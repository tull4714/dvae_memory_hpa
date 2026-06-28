import os
import sys
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# ============================================================
# dvae_curriculum.py에서 polynomial_tf, PhysicsBasisLayer 로드
# (HPA 시뮬레이션 + 물리 기저)
# ============================================================
import importlib.util, shutil

module_path = '/content/drive/MyDrive'
if module_path not in sys.path:
    sys.path.append(module_path)

_dvae_path = '/content/drive/MyDrive/NonlinearMemory/dvae_curriculum.py'
_pycache = '/content/drive/MyDrive/NonlinearMemory/__pycache__'
if os.path.exists(_pycache):
    shutil.rmtree(_pycache)
for _key in list(sys.modules.keys()):
    if 'dvae_curriculum' in _key:
        del sys.modules[_key]

_spec = importlib.util.spec_from_file_location("dvae_curriculum", _dvae_path)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

polynomial_tf     = _mod.polynomial_tf
PhysicsBasisLayer = _mod.PhysicsBasisLayer
print("polynomial_tf, PhysicsBasisLayer 로드 성공")


# ============================================================
# 유틸
# ============================================================
def normalize_with_rms(I_data, Q_data):
    magnitude = np.sqrt(I_data**2 + Q_data**2)
    rms = np.sqrt(np.mean(magnitude**2))
    if rms > 0:
        return I_data/rms, Q_data/rms, rms
    return I_data, Q_data, 1.0

def to_1ch(i, q):
    return np.stack([i, q], axis=-1).astype(np.float32)


# ============================================================
# Transformer DPD 구성 요소
# ============================================================
@tf.keras.utils.register_keras_serializable()
class PositionalEncoding(layers.Layer):
    def __init__(self, sequence_length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.pos_encoding = self._positional_encoding(sequence_length, d_model)

    def _get_angles(self, position, i, d_model):
        return position * (1 / tf.pow(10000.0,
                          (2 * (i // 2)) / tf.cast(d_model, tf.float32)))

    def _positional_encoding(self, position, d_model):
        angle_rads = self._get_angles(
            tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            tf.range(d_model, dtype=tf.float32)[tf.newaxis, :], d_model)
        sines   = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pe = tf.concat([sines, cosines], axis=-1)
        return tf.cast(pe[tf.newaxis, ...], tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

    def get_config(self):
        c = super().get_config()
        c.update({"sequence_length": self.sequence_length, "d_model": self.d_model})
        return c


def transformer_encoder(inputs, d_model, num_heads, ff_dim, dropout=0.1):
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inputs, inputs)
    attn = layers.Dropout(dropout)(attn)
    out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attn)
    ffn  = layers.Dense(ff_dim, activation="relu")(out1)
    ffn  = layers.Dense(d_model)(ffn)
    ffn  = layers.Dropout(dropout)(ffn)
    return layers.LayerNormalization(epsilon=1e-6)(out1 + ffn)


def build_transformer_dpd(seq_len, d_model, num_heads, ff_dim, num_layers,
                          k_orders=(2, 4), q_depth=3):
    """
    물리기저 + Transformer + residual 구조의 DPD 모델.

    구조:
      원본 IQ → PhysicsBasisLayer (메모리 다항식 기저)
              → Dense(d_model) → PositionalEncoding
              → Transformer 블록 × num_layers
              → Dense(2) = correction
      predistorted = 원본 + correction   (residual)

    residual 구조가 핵심: 모델은 작은 correction만 학습하면 되고,
    DLA loss = MSE(원본 - HPA(predistorted))로 추론 목표와 일치시킴.
    """
    inputs = layers.Input(shape=(seq_len, 2))

    # 물리 기저: (b, seq_len, 2) → (b, seq_len, 2+2KQ)
    basis = PhysicsBasisLayer(k_orders=k_orders, q_depth=q_depth)(inputs)

    x = layers.Dense(d_model)(basis)
    x = PositionalEncoding(seq_len, d_model)(x)
    for _ in range(num_layers):
        x = transformer_encoder(x, d_model, num_heads, ff_dim)

    correction = layers.Dense(2, activation="linear")(x)
    predistorted = layers.Add()([inputs, correction])   # residual

    return models.Model(inputs, predistorted, name="Transformer_DPD_DLA")


# ============================================================
# DLA 래퍼 모델 (train_step에서 HPA 통과)
# ============================================================
@tf.keras.utils.register_keras_serializable()
class TransformerDPD_DLA(models.Model):
    def __init__(self, core, backoff=1.0, memory_depth=0, snr_db=-1.0, **kwargs):
        super().__init__(**kwargs)
        self.core          = core
        self.backoff       = backoff
        self._memory_depth = memory_depth
        self._snr_db       = snr_db

    def set_curriculum(self, memory_depth, snr_db):
        self._memory_depth = memory_depth
        self._snr_db       = snr_db

    def call(self, inputs, training=False):
        return self.core(inputs, training=training)

    def compile(self, optimizer, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer
        self.loss_tracker = {"loss": tf.keras.metrics.Mean(name="loss")}

    @property
    def metrics(self):
        return list(self.loss_tracker.values())

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            predistorted = self.core(x, training=True)
            hpa_output = polynomial_tf(
                predistorted, backoff=self.backoff,
                snr_db=self._snr_db, memory_depth=self._memory_depth)
            loss = tf.reduce_mean(tf.square(x - hpa_output))
        grads = tape.gradient(loss, self.core.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.core.trainable_weights))
        self.loss_tracker["loss"].update_state(loss)
        return {k: v.result() for k, v in self.loss_tracker.items()}

    def test_step(self, data):
        x, y = data
        predistorted = self.core(x, training=False)
        hpa_output = polynomial_tf(
            predistorted, backoff=self.backoff,
            snr_db=self._snr_db, memory_depth=self._memory_depth)
        loss = tf.reduce_mean(tf.square(x - hpa_output))
        self.loss_tracker["loss"].update_state(loss)
        return {k: v.result() for k, v in self.loss_tracker.items()}


# ============================================================
# 하이퍼파라미터
# ============================================================
N          = 64
seq_len    = N
back_off   = 1            # ★ 데이터 생성과 동일 (16QAM)
batch_size = N * 2

d_model    = 64
num_heads  = 4
ff_dim     = 128
num_layers = 2
K_ORDERS   = (2, 4)
Q_DEPTH    = 3

# 커리큘럼: memory_depth 단계적 증가, 훈련 중 노이즈 없음
CURRICULUM_STAGES = {
    0: {"memory_depth": 0, "snr_db": -1.0, "total_epochs": 30, "init_lr": 1e-3},
    1: {"memory_depth": 1, "snr_db": -1.0, "total_epochs": 30, "init_lr": 5e-4},
    2: {"memory_depth": 2, "snr_db": -1.0, "total_epochs": 40, "init_lr": 2.5e-4},
}
SAVE_INTERVAL = 5
model_dir = '/content/drive/MyDrive/NonlinearMemory/'

# ============================================================
# 데이터 로드 (원본 신호만 — DLA)
# ============================================================
file_I = '/content/drive/MyDrive/input_target_I.csv'
file_Q = '/content/drive/MyDrive/input_target_Q.csv'

outputData_I = pd.read_csv(file_I, header=None).to_numpy()
outputData_Q = pd.read_csv(file_Q, header=None).to_numpy()
print(f"outputData_I shape: {outputData_I.shape}")

for nm, arr in [("I", outputData_I), ("Q", outputData_Q)]:
    if arr.shape[0] % seq_len != 0:
        raise ValueError(f"{nm} samples ({arr.shape[0]}) not divisible by {seq_len}")

outputData_I = outputData_I.reshape(-1, seq_len)
outputData_Q = outputData_Q.reshape(-1, seq_len)
len_i = outputData_I.shape[0]
print(f"총 시퀀스 수: {len_i}")

n_val   = int(0.1 * len_i)
n_train = int(0.8 * len_i)

train_I = outputData_I[:n_train];          train_Q = outputData_Q[:n_train]
val_I   = outputData_I[n_train:n_train+n_val]; val_Q = outputData_Q[n_train:n_train+n_val]
test_I  = outputData_I[n_train+n_val:];    test_Q  = outputData_Q[n_train+n_val:]

_, _, rms_train = normalize_with_rms(train_I, train_Q)
print(f"rms_train: {rms_train:.6f}")

X_in_train_ch = to_1ch(train_I/rms_train, train_Q/rms_train)
X_in_val_ch   = to_1ch(val_I/rms_train,   val_Q/rms_train)
X_in_test_ch  = to_1ch(test_I/rms_train,  test_Q/rms_train)
# y는 더미 (DLA에서 미사용)
print(f"X_in_train_ch.shape: {X_in_train_ch.shape}")

# ============================================================
# 모델 생성
# ============================================================
core = build_transformer_dpd(seq_len, d_model, num_heads, ff_dim, num_layers,
                             k_orders=K_ORDERS, q_depth=Q_DEPTH)
model = TransformerDPD_DLA(core, backoff=float(back_off),
                           memory_depth=CURRICULUM_STAGES[0]["memory_depth"],
                           snr_db=CURRICULUM_STAGES[0]["snr_db"])
model.compile(optimizer=tf.keras.optimizers.Adam(
    CURRICULUM_STAGES[0]["init_lr"], clipnorm=1.0))
model.build((None, seq_len, 2))
core.summary()

# ============================================================
# Gradient 확인 (DLA)
# ============================================================
print("\n--- Gradient 흐름 확인 ---")
sx = tf.constant(X_in_train_ch[:128])
with tf.GradientTape() as tape:
    pred = core(sx, training=True)
    hpa  = polynomial_tf(pred, backoff=float(back_off), snr_db=-1.0,
                         memory_depth=CURRICULUM_STAGES[0]["memory_depth"])
    lc   = tf.reduce_mean(tf.square(sx - hpa))
gc = tape.gradient(lc, core.trainable_variables)
gn = [tf.norm(g).numpy() for g in gc if g is not None]
print(f"gradient 최소: {min(gn):.2e}")
print(f"gradient 최대: {max(gn):.2e}")
print(f"gradient가 0인 레이어 수: {sum(1 for g in gn if g < 1e-10)}")
print("--------------------------\n")

# ============================================================
# EVM 모니터 (추론 EVM과 동일 정의)
# ============================================================
class EVMMonitor(tf.keras.callbacks.Callback):
    def __init__(self, sample_x, backoff, md_getter):
        self.sx = tf.constant(sample_x[:128])
        self.bo = backoff
        self.get_md = md_getter

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model.core(self.sx, training=False)
        md   = self.get_md()
        hpa  = polynomial_tf(pred, backoff=self.bo, snr_db=-1.0, memory_depth=md)
        den  = tf.reduce_mean(tf.square(self.sx))
        evm  = (tf.sqrt(tf.reduce_mean(tf.square(self.sx - hpa)) / den) * 100).numpy()
        hpa0 = polynomial_tf(self.sx, backoff=self.bo, snr_db=-1.0, memory_depth=md)
        evm0 = (tf.sqrt(tf.reduce_mean(tf.square(self.sx - hpa0)) / den) * 100).numpy()
        corr = pred - self.sx
        mag  = tf.reduce_mean(tf.abs(corr)).numpy()
        loss = logs.get('loss', 0); vl = logs.get('val_loss', 0)
        lr   = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        flag = "✓ 개선" if evm < evm0 else "✗ 악화"
        print(f"\n[Epoch {epoch+1}] correction={mag:.4f} | "
              f"EVM(DPD)={evm:.2f}% vs EVM(HPA)={evm0:.2f}% {flag} | "
              f"loss={loss:.2e} | val_loss={vl:.2e} | lr={lr:.2e}")

evm_monitor = EVMMonitor(X_in_train_ch, float(back_off), lambda: model._memory_depth)

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# ============================================================
# Resume 탐색
# ============================================================
def find_latest_ckpt(model_dir):
    pat = re.compile(r'transformer_dla_stage(\d+)_epoch(\d+)\.weights\.h5')
    cands = []
    for f in os.listdir(model_dir):
        m = pat.match(f)
        if m:
            cands.append((int(m.group(1)), int(m.group(2)), os.path.join(model_dir, f)))
    if not cands:
        return 0, 0, None
    cands.sort(key=lambda t: (t[0], t[1]))
    return cands[-1]

resume_stage, resume_epoch_in_stage, resume_path = find_latest_ckpt(model_dir)

if resume_path is not None:
    print(f"\n저장된 가중치 발견: {resume_path}")
    _ = core(tf.constant(X_in_train_ch[:N]), training=False)
    model.load_weights(resume_path)
    model.set_curriculum(
        memory_depth=CURRICULUM_STAGES[resume_stage]["memory_depth"],
        snr_db=CURRICULUM_STAGES[resume_stage]["snr_db"])
    print(f"훈련 재개: Stage {resume_stage}")
else:
    print("\n저장된 가중치 없음 — Stage 0부터 새로 훈련합니다.")

# ============================================================
# 커리큘럼 훈련 루프
# ============================================================
for stage_idx in range(resume_stage, len(CURRICULUM_STAGES)):
    cfg = CURRICULUM_STAGES[stage_idx]
    total_epochs = cfg["total_epochs"]
    stage_init_lr = cfg["init_lr"]

    model.set_curriculum(memory_depth=cfg["memory_depth"], snr_db=cfg["snr_db"])
    print(f"\n현재 Stage {stage_idx} | memory={cfg['memory_depth']}tap")

    model.optimizer.learning_rate = stage_init_lr
    print(f"LR 초기화: {stage_init_lr:.2e}")

    early_stopping.best = np.inf; early_stopping.wait = 0
    early_stopping.stopped_epoch = 0
    lr_scheduler.best = np.inf; lr_scheduler.wait = 0
    lr_scheduler.cooldown_counter = 0

    epoch_start = resume_epoch_in_stage if stage_idx == resume_stage else 0
    resume_epoch_in_stage = 0

    if epoch_start >= total_epochs:
        print(f"Stage {stage_idx} 이미 완료 — 다음으로")
        continue

    save_points = list(range(((epoch_start // SAVE_INTERVAL) + 1) * SAVE_INTERVAL,
                             total_epochs + 1, SAVE_INTERVAL))
    if not save_points or save_points[-1] < total_epochs:
        save_points.append(total_epochs)

    for save_epoch in save_points:
        epochs_to_run = save_epoch - epoch_start
        if epochs_to_run <= 0:
            continue
        print(f"\n[Stage {stage_idx}] {epoch_start+1}~{save_epoch} epoch "
              f"(memory={cfg['memory_depth']}tap, LR_init={stage_init_lr:.2e})")

        model.fit(X_in_train_ch, X_in_train_ch,
                  validation_data=(X_in_val_ch, X_in_val_ch),
                  epochs=epochs_to_run, batch_size=batch_size,
                  callbacks=[evm_monitor, lr_scheduler, early_stopping],
                  verbose=1)

        sp = os.path.join(model_dir,
                          f"transformer_dla_stage{stage_idx}_epoch{save_epoch}.weights.h5")
        model.save_weights(sp)
        print(f"저장 완료: {sp}")
        print(f"Test 평가 (Stage {stage_idx}, epoch {save_epoch}):")
        model.evaluate(X_in_test_ch, X_in_test_ch, verbose=1)

        if early_stopping.stopped_epoch > 0:
            print(f"Early stopping — Stage {stage_idx} 종료")
            break
        epoch_start = save_epoch

# ============================================================
# 최종 평가
# ============================================================
print("\n" + "="*55)
print("훈련 완료 — 최종 test 평가")
print("="*55)
model.evaluate(X_in_test_ch, X_in_test_ch, verbose=1)

# 최종 EVM
sx = tf.constant(X_in_test_ch[:2000])
pred = core(sx, training=False)
hpa  = polynomial_tf(pred, backoff=float(back_off), snr_db=-1.0, memory_depth=2)
hpa0 = polynomial_tf(sx,   backoff=float(back_off), snr_db=-1.0, memory_depth=2)
den  = tf.reduce_mean(tf.square(sx))
evm  = (tf.sqrt(tf.reduce_mean(tf.square(sx - hpa)) / den) * 100).numpy()
evm0 = (tf.sqrt(tf.reduce_mean(tf.square(sx - hpa0)) / den) * 100).numpy()
print(f"\n최종 EVM (보상 없음, HPA): {evm0:.4f}%")
print(f"최종 EVM (Transformer DPD): {evm:.4f}%")
print(f"{'✓ DPD 성공' if evm < evm0 else '✗ DPD 실패'}")
