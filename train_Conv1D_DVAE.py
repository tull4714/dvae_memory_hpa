import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import re

# dvae_curriculum.py를 파일 경로로 직접 로드
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

DVAE_Curriculum     = _mod.DVAE_Curriculum
CurriculumScheduler = _mod.CurriculumScheduler
PhysicsBasisLayer   = _mod.PhysicsBasisLayer
polynomial_tf       = _mod.polynomial_tf
print("dvae_curriculum 직접 로드 성공")

# -----------------------------
# RMS Normalization
# -----------------------------
def normalize_with_rms(I_data, Q_data):
    magnitude = np.sqrt(I_data**2 + Q_data**2)
    rms = np.sqrt(np.mean(magnitude**2))
    if rms > 0:
        return I_data/rms, Q_data/rms, rms
    return I_data, Q_data, 1.0

def to_1ch(i, q):
    return np.stack([i, q], axis=-1).astype(np.float32)

def to_complex(x2ch):
    return x2ch[..., 0] + 1j * x2ch[..., 1]


# ============================================================
# TCN + 물리 기저 결합 인코더
#
# 원본 IQ → PhysicsBasisLayer (Volterra 기저 생성)
#         → Dilated Conv1D 스택 (dilation 1,2,4,8)
#         → z (batch, seq_len, 64)
#
# dilation으로 receptive field를 메모리 길이보다 넓게 확보:
#   1+2+4+8 = 15탭 → HPA 메모리(q=2)를 충분히 커버
# ============================================================
def build_encoder(seq_len, k_orders=(2, 4), q_depth=3):
    inputs = layers.Input(shape=(seq_len, 2))

    # 물리 기저 생성: (batch, seq_len, 2) → (batch, seq_len, 2+2*K*Q)
    x = PhysicsBasisLayer(k_orders=k_orders, q_depth=q_depth)(inputs)

    # Dilated TCN 스택 (causal padding으로 인과성 유지)
    x = layers.Conv1D(64,  3, padding="causal", dilation_rate=1, activation="relu")(x)
    x = layers.Conv1D(64,  3, padding="causal", dilation_rate=2, activation="relu")(x)
    x = layers.Conv1D(128, 3, padding="causal", dilation_rate=4, activation="relu")(x)
    x = layers.Conv1D(128, 3, padding="causal", dilation_rate=8, activation="relu")(x)

    z = layers.Conv1D(64, 1, padding="same")(x)   # (batch, seq_len, 64)
    return models.Model(inputs, z, name="tcn_physics_encoder")


# ============================================================
# TCN 디코더 (Dilated Conv1D)
# 입력: (batch, seq_len, 64) → 출력: (batch, seq_len, 2) correction
# ============================================================
def build_decoder(seq_len, channels=2, latent_dim=64):
    inputs = layers.Input(shape=(seq_len, latent_dim))
    x = layers.Conv1D(128, 3, padding="causal", dilation_rate=1, activation="relu")(inputs)
    x = layers.Conv1D(64,  3, padding="causal", dilation_rate=2, activation="relu")(x)
    x = layers.Conv1D(64,  3, padding="causal", dilation_rate=4, activation="relu")(x)
    outputs = layers.Conv1D(channels, 1, padding="same")(x)
    return models.Model(inputs, outputs, name="tcn_decoder")


# ============================================================
# 커리큘럼 학습 설정
#
# DLA 방식: train_step에서 polynomial_tf(HPA)를 직접 통과.
# snr_db=-1.0 → 훈련 중 HPA에 노이즈 없음 (타겟 안정화).
# memory_depth가 단계적으로 증가하며 HPA 메모리 차수를 늘림.
# ============================================================
CURRICULUM_STAGES = {
    0: {"memory_depth": 0, "snr_db": -1.0, "total_epochs": 30, "init_lr": 1e-3},
    1: {"memory_depth": 1, "snr_db": -1.0, "total_epochs": 30, "init_lr": 5e-4},
    2: {"memory_depth": 2, "snr_db": -1.0, "total_epochs": 40, "init_lr": 2.5e-4},
}
SAVE_INTERVAL = 5


def find_latest_curriculum_checkpoint(model_dir):
    pattern    = re.compile(r'curriculum_stage(\d+)_epoch(\d+)\.weights\.h5')
    candidates = []
    for fname in os.listdir(model_dir):
        m = pattern.match(fname)
        if m:
            candidates.append(
                (int(m.group(1)), int(m.group(2)),
                 os.path.join(model_dir, fname))
            )
    if not candidates:
        return 0, 0, None
    candidates.sort(key=lambda t: (t[0], t[1]))
    return candidates[-1]


# ============================================================
# 하이퍼파라미터
# ============================================================
N          = 64
seq_len    = N
back_off   = 1    # ★ 데이터 생성과 동일하게 1로 설정 (16QAM)
batch_size = N * 2

# 물리 기저 설정 (PhysicsBasisLayer)
K_ORDERS = (2, 4)   # 3차, 5차 비선형
Q_DEPTH  = 3        # 메모리 깊이 q=0,1,2

file2_I = '/content/drive/MyDrive/input_target_I.csv'
file2_Q = '/content/drive/MyDrive/input_target_Q.csv'

# ============================================================
# 데이터 로드 (DLA: 원본 신호만 필요)
# input_iq(HPA 출력) CSV는 사용하지 않음
# — train_step에서 polynomial_tf로 HPA를 직접 통과시키기 때문
# ============================================================
outputData_I = pd.read_csv(file2_I, header=None).to_numpy()
outputData_Q = pd.read_csv(file2_Q, header=None).to_numpy()

print(f"outputData_I shape: {outputData_I.shape}")

for name, arr in [("I output", outputData_I), ("Q output", outputData_Q)]:
    if arr.shape[0] % seq_len != 0:
        raise ValueError(
            f"Total samples for {name} ({arr.shape[0]}) must be "
            f"divisible by seq_len ({seq_len})."
        )

outputData_I = outputData_I.reshape(-1, seq_len)
outputData_Q = outputData_Q.reshape(-1, seq_len)

len_i = outputData_I.shape[0]
print(f"총 시퀀스 수: {len_i}")

# ============================================================
# Train / Val / Test split (원본 신호)
# ============================================================
n_val   = int(0.1 * len_i)
n_train = int(0.8 * len_i)

outputData_train_I = outputData_I[0:n_train]
outputData_train_Q = outputData_Q[0:n_train]
outputData_val_I   = outputData_I[n_train: n_train + n_val]
outputData_val_Q   = outputData_Q[n_train: n_train + n_val]
outputData_test_I  = outputData_I[n_train + n_val:]
outputData_test_Q  = outputData_Q[n_train + n_val:]

# ============================================================
# 정규화 (원본 신호 기준 RMS)
# ============================================================
_, _, rms_train = normalize_with_rms(outputData_train_I, outputData_train_Q)
print(f"rms_train: {rms_train:.6f}")

target_I_train = outputData_train_I / rms_train
target_Q_train = outputData_train_Q / rms_train
target_I_val   = outputData_val_I   / rms_train
target_Q_val   = outputData_val_Q   / rms_train
target_I_test  = outputData_test_I  / rms_train
target_Q_test  = outputData_test_Q  / rms_train

# ============================================================
# DLA 데이터 구성 (실제 HPA 통과 방식)
#
# X_in = 원본 신호만 필요.
# train_step이 predistorted를 polynomial_tf(HPA)에 직접 통과시켜
# loss = MSE(원본 - HPA(predistorted))를 계산하므로
# correction_target(원본 - HPA(원본))과 input_iq CSV는 불필요.
#
# y 자리에는 형식상 X_in을 그대로 전달 (train_step에서 사용 안 함).
# ============================================================
X_in_train_ch  = to_1ch(target_I_train, target_Q_train)
X_in_val_ch    = to_1ch(target_I_val,   target_Q_val)
X_in_test_ch   = to_1ch(target_I_test,  target_Q_test)

# y 더미 (train_step/test_step에서 사용하지 않음)
X_out_train_ch = X_in_train_ch
X_out_val_ch   = X_in_val_ch
X_out_test_ch  = X_in_test_ch

print(f"X_in_train_ch.shape : {X_in_train_ch.shape}")
print(f"원본 신호 RMS (train): "
      f"{np.sqrt(np.mean(target_I_train**2 + target_Q_train**2)):.6f}")

# ============================================================
# 모델 생성 (TCN + 물리 기저)
# ============================================================
encoder = build_encoder(seq_len, k_orders=K_ORDERS, q_depth=Q_DEPTH)
decoder = build_decoder(seq_len)

dvae = DVAE_Curriculum(
    encoder      = encoder,
    decoder      = decoder,
    backoff      = float(back_off),
    memory_depth = CURRICULUM_STAGES[0]["memory_depth"],
    snr_db       = CURRICULUM_STAGES[0]["snr_db"],
)
dvae.compile(optimizer=tf.keras.optimizers.Adam(
    CURRICULUM_STAGES[0]["init_lr"], clipnorm=1.0))
dvae.build((None, seq_len, 2))
dvae.encoder.summary()
dvae.decoder.summary()

# ============================================================
# Gradient 확인 (DLA: HPA 통과 loss로 검증)
# ============================================================
print("\n--- Gradient 흐름 확인 ---")
sample_x = tf.constant(X_in_train_ch[:128])
with tf.GradientTape() as tape:
    pred       = dvae(sample_x, training=True)
    hpa_out    = polynomial_tf(pred, backoff=float(back_off),
                               snr_db=-1.0,
                               memory_depth=CURRICULUM_STAGES[0]["memory_depth"])
    loss_check = tf.reduce_mean(tf.square(sample_x - hpa_out))
grads_check = tape.gradient(loss_check, dvae.trainable_variables)
grad_norms  = [tf.norm(g).numpy() for g in grads_check if g is not None]
print(f"gradient 최소: {min(grad_norms):.2e}")
print(f"gradient 최대: {max(grad_norms):.2e}")
print(f"gradient가 0인 레이어 수: {sum(1 for g in grad_norms if g < 1e-10)}")
print("--------------------------\n")

# ============================================================
# 콜백 정의
# ============================================================
class CorrectionMonitor(tf.keras.callbacks.Callback):
    """
    DLA 방식 모니터:
      매 epoch 끝에 predistorted를 실제 HPA에 통과시켜
      EVM = ||원본 - HPA(predistorted)|| / ||원본|| 을 출력.
      이 EVM이 추론 시 측정하는 EVM과 동일 → 학습 진척을 직접 확인.
    """
    def __init__(self, sample_x, backoff, memory_depth_getter):
        self.sample_x = tf.constant(sample_x[:128])
        self.backoff  = backoff
        self.get_md   = memory_depth_getter   # 현재 memory_depth 조회 함수
        self.history  = []

    def on_epoch_end(self, epoch, logs=None):
        pred       = self.model(self.sample_x, training=False)
        correction = pred - self.sample_x
        mag        = tf.reduce_mean(tf.abs(correction)).numpy()

        # 실제 HPA 통과 후 EVM (추론 EVM과 동일 정의)
        md      = self.get_md()
        hpa_out = polynomial_tf(pred, backoff=self.backoff,
                                snr_db=-1.0, memory_depth=md)
        num = tf.reduce_mean(tf.square(self.sample_x - hpa_out))
        den = tf.reduce_mean(tf.square(self.sample_x))
        evm = (tf.sqrt(num / den) * 100.0).numpy()

        # 보상 없을 때 EVM (참고용)
        hpa_noc = polynomial_tf(self.sample_x, backoff=self.backoff,
                                snr_db=-1.0, memory_depth=md)
        num0 = tf.reduce_mean(tf.square(self.sample_x - hpa_noc))
        evm0 = (tf.sqrt(num0 / den) * 100.0).numpy()

        loss     = logs.get('loss',     0)
        val_loss = logs.get('val_loss', 0)
        lr       = float(tf.keras.backend.get_value(
                       self.model.optimizer.learning_rate))
        self.history.append(evm)
        flag = "✓ 개선" if evm < evm0 else "✗ 악화"
        print(f"\n[Epoch {epoch+1}] "
              f"correction={mag:.4f} | "
              f"EVM(DVAE)={evm:.2f}% vs EVM(HPA)={evm0:.2f}% {flag} | "
              f"loss={loss:.2e} | val_loss={val_loss:.2e} | lr={lr:.2e}")

correction_monitor = CorrectionMonitor(
    X_in_train_ch, float(back_off), lambda: dvae._memory_depth)

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1
)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
)

# ============================================================
# Resume
# ============================================================
model_dir = '/content/drive/MyDrive/NonlinearMemory/'
scheduler = CurriculumScheduler()

resume_stage, resume_epoch_in_stage, resume_path = \
    find_latest_curriculum_checkpoint(model_dir)

if resume_path is not None:
    print(f"\n저장된 가중치 발견: {resume_path}")
    if hasattr(dvae.optimizer, 'build') and dvae.trainable_variables:
        dvae.optimizer.build(dvae.trainable_variables)
    dvae.load_weights(resume_path)
    for _ in range(resume_stage):
        scheduler.advance()
    dvae.set_curriculum(
        memory_depth = CURRICULUM_STAGES[resume_stage]["memory_depth"],
        snr_db       = CURRICULUM_STAGES[resume_stage]["snr_db"],
    )
    print(f"훈련 재개: {scheduler.description}")
else:
    print("\n저장된 가중치 없음 — Stage 0부터 새로 훈련합니다.")

# ============================================================
# 커리큘럼 훈련 루프
# ============================================================
for stage_idx in range(resume_stage, len(CURRICULUM_STAGES)):
    cfg           = CURRICULUM_STAGES[stage_idx]
    total_epochs  = cfg["total_epochs"]
    stage_init_lr = cfg["init_lr"]

    dvae.set_curriculum(
        memory_depth = cfg["memory_depth"],
        snr_db       = cfg["snr_db"],
    )
    scheduler.status()

    # Stage 시작 시 LR 리셋 (Keras 3.x 호환)
    dvae.optimizer.learning_rate = stage_init_lr
    print(f"LR 초기화: {stage_init_lr:.2e}")

    early_stopping.best           = np.inf
    early_stopping.wait           = 0
    early_stopping.stopped_epoch  = 0
    lr_scheduler.best             = np.inf
    lr_scheduler.wait             = 0
    lr_scheduler.cooldown_counter = 0

    epoch_start           = resume_epoch_in_stage if stage_idx == resume_stage else 0
    resume_epoch_in_stage = 0

    if epoch_start >= total_epochs:
        print(f"Stage {stage_idx}는 이미 완료 — 다음 단계로 넘어갑니다.")
        if not scheduler.is_final_stage:
            scheduler.advance()
        continue

    save_points = list(range(
        ((epoch_start // SAVE_INTERVAL) + 1) * SAVE_INTERVAL,
        total_epochs + 1,
        SAVE_INTERVAL
    ))
    if not save_points or save_points[-1] < total_epochs:
        save_points.append(total_epochs)

    for save_epoch in save_points:
        epochs_to_run = save_epoch - epoch_start
        if epochs_to_run <= 0:
            continue

        print(f"\n[Stage {stage_idx}] "
              f"{epoch_start+1}~{save_epoch} epoch 훈련 중 "
              f"(memory={cfg['memory_depth']}tap, "
              f"SNR={cfg['snr_db']}dB, LR_init={stage_init_lr:.2e})")

        dvae.fit(
            x               = X_in_train_ch,
            y               = X_out_train_ch,
            validation_data = (X_in_val_ch, X_out_val_ch),
            epochs          = epochs_to_run,
            batch_size      = batch_size,
            callbacks       = [correction_monitor, lr_scheduler, early_stopping],
            verbose         = 1
        )

        save_name = f"curriculum_stage{stage_idx}_epoch{save_epoch}.weights.h5"
        save_path = os.path.join(model_dir, save_name)
        dvae.save_weights(save_path)
        print(f"저장 완료: {save_path}")

        print(f"Test set 평가 (Stage {stage_idx}, epoch {save_epoch}):")
        dvae.evaluate(X_in_test_ch, X_out_test_ch, verbose=1)

        if early_stopping.stopped_epoch > 0:
            print(f"Early stopping 발동 — Stage {stage_idx} 종료")
            break

        epoch_start = save_epoch

    if not scheduler.is_final_stage:
        scheduler.advance()

# ============================================================
# 최종 평가
# ============================================================
print("\n" + "="*55)
print("커리큘럼 훈련 완료 — 최종 test set 평가")
print("="*55)
dvae.evaluate(X_in_test_ch, X_out_test_ch, verbose=1)

print("\npredistorted 신호 생성 중...")
pred_list = []
for i in range(0, len(X_in_test_ch), batch_size):
    batch        = tf.constant(X_in_test_ch[i:i+batch_size])
    predistorted = dvae(batch, training=False)
    pred_list.append(predistorted.numpy())
pred_test = np.concatenate(pred_list, axis=0)
print(f"pred_test.shape: {pred_test.shape}")
