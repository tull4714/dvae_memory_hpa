"""
일반적인 Transformer DPD 학습 (문헌 기준선)

VT-AE 논문에서 "기존 방법 대비" 비교에 쓰는 기준선.  [13]의 Transformer DPD처럼
  - 입력: I, Q (+ 진폭 |x|) 시퀀스 (메모리 다항식 기저 없음)
  - 출력: 전치왜곡 신호를 직접 출력 (잔차 연결 없음)
  - 커리큘럼 없음
  - 학습: EXP_CONV_TRAIN_MODE
      'dla' (기본) — 직접 학습. loss = mean|G·x − PA(DPD(x))|^2  (VT-AE와 같은 손실, 구조만 다름)
      'ila'       — 간접 학습. 반복 i마다
                     z = DPD(x)  (첫 반복은 z = x),  y = PA(z),
                     후치왜곡기 P가 y/G → z 를 근사하도록 MSE로 학습하고, 학습된 P를 DPD로 복사
                     (5차 PA, IBO 8 dB 시험에서 반복할수록 발산 — 포화 영역에서 후치 역함수가 정의되지 않음)
구조·입력·학습 방식은 dvae_curriculum.py의 EXP_CONV_* 설정에서 바꾼다.
PA 차수, IBO, 목표 이득 G, 배치, 데이터 비율, 전체 epoch 수(커리큘럼 단계 epoch의 합)는
VT-AE / Transformer 학습 파일과 같은 값을 쓴다.

체크포인트: transformer_conv_<설정>_stage0_epoch<누적 epoch>.weights.h5  (Keras 모델 core의 가중치)
"""
import os
import sys
import re
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
import importlib.util, shutil

# ============================================================
# 경로 설정, dvae_curriculum.py 로드
# ============================================================
BASE_DIR  = '/content/drive/MyDrive/NonlinearMemory'
MODEL_DIR = os.path.join(BASE_DIR, 'vt-ae')

_dvae_path = os.path.join(BASE_DIR, 'dvae_curriculum.py')
_pycache   = os.path.join(BASE_DIR, '__pycache__')
if os.path.exists(_pycache):
    shutil.rmtree(_pycache)
for _key in list(sys.modules.keys()):
    if 'dvae_curriculum' in _key:
        del sys.modules[_key]
_spec = importlib.util.spec_from_file_location("dvae_curriculum", _dvae_path)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
polynomial_tf = _mod.polynomial_tf
polynomial_np = _mod.polynomial_np
print(f"실험 설정: {_mod.exp_tag()}, A_SAT {_mod.A_SAT:.3f}, 목표 이득 G {_mod.G_FIX}")

if _mod.EXP_SEED is not None:      # 반복 학습: 가중치 초기값·미니배치 순서·dropout을 seed로 고정
    tf.keras.utils.set_random_seed(_mod.EXP_SEED)
print(f"학습 seed: {_mod.EXP_SEED if _mod.EXP_SEED is not None else '고정 안 함 (1회차)'}")


# ============================================================
# 하이퍼파라미터 (dvae_curriculum.py의 실험 설정을 따름)
# ============================================================
seq_len       = 64
back_off      = float(_mod.PA_IBO_DB)
G             = float(_mod.G_FIX)
batch_size    = _mod.EXP_BATCH_SIZE
DATA_FRACTION = _mod.EXP_DATA_FRACTION
TOTAL_EPOCHS  = _mod.total_epochs()          # VT-AE와 같은 전체 epoch 수 (커리큘럼 단계 합)
SAVE_INTERVAL = _mod.EXP_SAVE_INTERVAL
TRAIN_MODE    = _mod.EXP_CONV_TRAIN_MODE     # 'ila' 또는 'dla'
INPUT_MODE    = _mod.EXP_CONV_INPUT          # 'iq' 또는 'iq_amp'
ILA_ITERS     = _mod.EXP_CONV_ILA_ITERS
INIT_LR       = 1e-3
REF_POWER     = 1.0                          # 정규화된 원신호의 평균 전력 = PA 기준 전력
CKPT_PREFIX   = _mod.ckpt_prefix('transformer_conv')
HISTORY_CSV   = os.path.join(MODEL_DIR, f'training_history_{CKPT_PREFIX}.csv')
print(f"체크포인트 접두어: {CKPT_PREFIX}, 학습 {TRAIN_MODE}, 입력 {INPUT_MODE}, "
      f"배치 {batch_size}, 데이터 비율 {DATA_FRACTION}, 전체 epoch {TOTAL_EPOCHS}")


# ============================================================
# 데이터 로드 (원신호, 물리적 시간 순서의 연속 스트림)
# ============================================================
xi = pd.read_csv(os.path.join(BASE_DIR, 'input_target_I.csv'), header=None).to_numpy().reshape(-1)
xq = pd.read_csv(os.path.join(BASE_DIR, 'input_target_Q.csv'), header=None).to_numpy().reshape(-1)
n_seq = len(xi) // seq_len
if DATA_FRACTION < 1.0:                       # 학습 예산 제한: 앞부분 연속 구간만 사용
    n_seq = int(DATA_FRACTION * n_seq)
x_all = (xi + 1j * xq)[:n_seq * seq_len]
n_train, n_val = int(0.8 * n_seq), int(0.1 * n_seq)
rms = np.sqrt(np.mean(np.abs(x_all[:n_train * seq_len]) ** 2))
x_all = x_all / rms                           # 학습 구간 평균 전력 = 1
x_tr = x_all[:n_train * seq_len]
x_va = x_all[n_train * seq_len:(n_train + n_val) * seq_len]
print(f"시퀀스 수: train {n_train}, val {n_val}")

def to_seq(c):                                 # 복소 스트림 → (시퀀스, 64, 2)
    s = c.reshape(-1, seq_len)
    return np.stack([s.real, s.imag], -1).astype(np.float32)

def from_seq(a):                               # (시퀀스, 64, 2) → 복소 스트림
    return (a[..., 0] + 1j * a[..., 1]).reshape(-1)

def pa_stream(z):                              # 연속 스트림으로 PA 통과 (메모리가 시퀀스 경계를 넘어감)
    return polynomial_np(z, back_off, ref_power=REF_POWER)


# ============================================================
# 모델
# ============================================================
core = _mod.build_transformer_conventional(seq_len, INPUT_MODE)
print(f"총 파라미터 수: {core.count_params():,}")

def dpd_stream(x):
    a = to_seq(x)
    y = np.concatenate([core(tf.constant(a[i:i + 4096]), training=False).numpy()
                        for i in range(0, len(a), 4096)])
    return from_seq(y)

def evm_gain_norm(ref, out):
    g = np.vdot(ref, out) / np.vdot(ref, ref)
    return np.sqrt(np.mean(np.abs(out / g - ref) ** 2) / np.mean(np.abs(ref) ** 2)) * 100, abs(g)

x_mon = x_va[:512 * seq_len]                   # 학습 곡선 기록용 검증 신호
evm_hpa_mon, g_hpa_mon = evm_gain_norm(x_mon, pa_stream(x_mon))

class History(tf.keras.callbacks.Callback):
    """매 epoch: 전체 PA(Q = 2) 기준 DPD 이득 정규화 EVM (시간 영역) 기록, 주기적으로 가중치 저장"""
    def __init__(self):
        super().__init__()
        self.offset = 0
        if not os.path.exists(HISTORY_CSV):
            with open(HISTORY_CSV, 'w', newline='') as f:
                csv.writer(f).writerow(['global_epoch', 'mode', 'loss', 'val_loss', 'lr',
                                        'evm_dpd_gn_pct', 'evm_hpa_gn_pct', 'gain_dpd'])
    def on_epoch_end(self, epoch, logs=None):
        ep = epoch + 1                          # fit의 initial_epoch를 누적 epoch로 사용
        evm, g = evm_gain_norm(x_mon, pa_stream(dpd_stream(x_mon)))
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        loss, vloss = logs.get('loss', np.nan), logs.get('val_loss', np.nan)
        with open(HISTORY_CSV, 'a', newline='') as f:
            csv.writer(f).writerow([ep, TRAIN_MODE, f'{loss:.6e}', f'{vloss:.6e}', f'{lr:.3e}',
                                    f'{evm:.4f}', f'{evm_hpa_mon:.4f}', f'{g:.4f}'])
        print(f"\n[{TRAIN_MODE}, epoch {ep}] EVM(Q=2, 이득 정규화) DPD {evm:.2f}% (|g| {g:.4f}) "
              f"vs HPA {evm_hpa_mon:.2f}% | loss {loss:.2e} | val_loss {vloss:.2e} | lr {lr:.2e}")
        if ep % SAVE_INTERVAL == 0 or ep == TOTAL_EPOCHS:
            path = os.path.join(MODEL_DIR, f"{CKPT_PREFIX}_stage0_epoch{ep}.weights.h5")
            core.save_weights(path)
            print(f"저장: {path}")

hist_cb = History()
lr_cb   = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                                               min_lr=1e-5, verbose=1)


# ============================================================
# 이어서 학습 (같은 접두어의 마지막 체크포인트)
# ============================================================
pat = re.compile(re.escape(CKPT_PREFIX) + r'_stage0_epoch(\d+)\.weights\.h5')
done = sorted([(int(m.group(1)), f) for f in os.listdir(MODEL_DIR) if (m := pat.fullmatch(f))])
start_epoch = 0
if done:
    start_epoch, fname = done[-1]
    _ = core(tf.zeros((1, seq_len, 2)))
    core.load_weights(os.path.join(MODEL_DIR, fname))
    print(f"저장된 가중치 발견: {fname} → epoch {start_epoch}부터 이어서 학습")
if start_epoch >= TOTAL_EPOCHS:
    print("이미 학습 완료"); sys.exit(0)


# ============================================================
# 학습
# ============================================================
if TRAIN_MODE == 'ila':
    # 간접 학습: 반복마다 현재 DPD로 PA를 구동해 (PA 출력/G → PA 입력) 쌍을 만들고 후치왜곡기를 학습
    core.compile(optimizer=tf.keras.optimizers.Adam(INIT_LR, clipnorm=1.0), loss='mse')
    ep_per_iter = int(np.ceil(TOTAL_EPOCHS / ILA_ITERS))
    for it in range(ILA_ITERS):
        e0, e1 = it * ep_per_iter, min((it + 1) * ep_per_iter, TOTAL_EPOCHS)
        if e1 <= start_epoch:
            continue
        first = (it == 0 and start_epoch == 0)
        z_tr = x_tr if first else dpd_stream(x_tr)     # 첫 반복은 DPD 없이 (z = x)
        z_va = x_va if first else dpd_stream(x_va)
        u_tr, u_va = pa_stream(z_tr) / G, pa_stream(z_va) / G   # 후치왜곡기 입력 = PA 출력 / 목표 이득
        print(f"\n[ILA 반복 {it + 1}/{ILA_ITERS}] epoch {max(e0, start_epoch) + 1}~{e1}, "
              f"PA 입력 z는 {'원신호' if first else '현재 DPD 출력'}")
        core.fit(to_seq(u_tr), to_seq(z_tr), validation_data=(to_seq(u_va), to_seq(z_va)),
                 initial_epoch=max(e0, start_epoch), epochs=e1, batch_size=batch_size,
                 callbacks=[hist_cb, lr_cb], verbose=1)
else:
    # 직접 학습: PA 모델을 거쳐 역전파 (VT-AE와 같은 손실)
    class ConvDLA(tf.keras.Model):
        def __init__(self, core_model):
            super().__init__()
            self.core = core_model
            self.tracker = tf.keras.metrics.Mean(name="loss")
        @property
        def metrics(self):
            return [self.tracker]
        def call(self, x, training=False):
            return self.core(x, training=training)
        def _loss(self, x, training):
            y = polynomial_tf(self.core(x, training=training), backoff=back_off,
                              memory_depth=2, ref_power=REF_POWER)
            return tf.reduce_mean(tf.square(G * x - y))
        def train_step(self, data):
            x, _ = data
            with tf.GradientTape() as tape:
                loss = self._loss(x, True)
            grads = tape.gradient(loss, self.core.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.core.trainable_weights))
            self.tracker.update_state(loss)
            return {"loss": self.tracker.result()}
        def test_step(self, data):
            x, _ = data
            self.tracker.update_state(self._loss(x, False))
            return {"loss": self.tracker.result()}

    net = ConvDLA(core)
    net.compile(optimizer=tf.keras.optimizers.Adam(INIT_LR, clipnorm=1.0))
    X_tr, X_va = to_seq(x_tr), to_seq(x_va)
    net.fit(X_tr, X_tr, validation_data=(X_va, X_va), initial_epoch=start_epoch, epochs=TOTAL_EPOCHS,
            batch_size=batch_size, callbacks=[hist_cb, lr_cb], verbose=1)

print(f"\n학습 완료. 학습 곡선 기록: {HISTORY_CSV}")
