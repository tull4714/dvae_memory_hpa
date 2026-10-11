"""
VT-AE DPD 학습 (DLA + 커리큘럼)

[이번 변경] 실험 환경 통합 + 커리큘럼 버그 수정
   1) 실험 설정은 모두 dvae_curriculum.py에서 읽는다:
      PA_ORDER(5/7/9), PA_IBO_DB, 목표 이득 G_FIX, EXP_BATCH_SIZE, EXP_DATA_FRACTION,
      EXP_USE_CURRICULUM, 단계 구성(EXP_STAGES_*), EXP_SAVE_INTERVAL.
      VT-AE와 Transformer 학습 파일이 같은 값을 쓰므로 조건이 자동으로 같아진다.
   2) 체크포인트 이름에 설정이 들어간다 (예: vtae_pa9_ibo8_b512_f0.02_cur_stage2_epoch20.weights.h5).
   3) [버그 수정] 커리큘럼 단계를 바꿀 때 컴파일된 train/test 함수를 버려 새 memory depth로
      다시 컴파일한다. 이전 코드는 2·3단계도 실제로는 Q = 0 PA로 학습되었다.
   4) EXP_DATA_FRACTION < 1이면 학습 데이터의 앞부분 일부만 사용한다 (학습 예산 제한 실험).

[변경 사항]
1) HPA 모델: dvae_curriculum.py의 표준 memory polynomial, 고정 기준 전력(ref_power = 1.0)
   학습 데이터는 원신호 RMS로 정규화되어 평균 전력이 정확히 1이므로 PA 기준 전력도 1.0으로 고정.
2) 단계마다 fit을 한 번만 호출 (initial_epoch 사용).
   이전에는 5 epoch씩 나눠 fit을 반복 호출해서 ReduceLROnPlateau/EarlyStopping의 상태가
   호출마다 초기화되었고, 논문에 쓴 "5 epoch 정체 시 학습률 절반, 10 epoch 정체 시 조기 종료"가
   실제로는 그렇게 동작하지 않았음. 체크포인트는 PeriodicSaver 콜백이 5 epoch마다 저장.
3) 학습 곡선 기록: 매 epoch 검증 데이터로 전체 모델(Q = 2) 기준 EVM을 계산해 CSV로 저장
   (단계가 달라도 같은 잣대로 비교 가능 → 커리큘럼 유무 비교 그림용).
4) USE_CURRICULUM = False 로 두면 처음부터 Q = 2로 학습 (커리큘럼 비교 실험용).
"""
import os
import sys
import re
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import importlib.util, shutil

# ============================================================
# 경로 설정
# ============================================================
BASE_DIR  = '/content/drive/MyDrive'
MODEL_DIR = os.path.join(BASE_DIR, 'NonlinearMemory')

# dvae_curriculum.py를 파일 경로로 직접 로드 (패키지/캐시 문제 회피)
_dvae_path = os.path.join(MODEL_DIR, 'dvae_curriculum.py')
_pycache   = os.path.join(MODEL_DIR, '__pycache__')
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
PA_IBO_DB           = _mod.PA_IBO_DB
dla_loss            = _mod.dla_loss
print(f"실험 설정: {_mod.exp_tag()}, A_SAT {_mod.A_SAT:.3f}, 목표 이득 G {_mod.G_FIX}")
print("dvae_curriculum 직접 로드 성공")


# ============================================================
# 유틸
# ============================================================
def normalize_with_rms(I_data, Q_data):
    rms = np.sqrt(np.mean(I_data**2 + Q_data**2))
    return (I_data / rms, Q_data / rms, rms) if rms > 0 else (I_data, Q_data, 1.0)

def to_1ch(i, q):
    return np.stack([i, q], axis=-1).astype(np.float32)


# ============================================================
# VT-AE 구조 (Volterra 기저 + 인과 dilated TCN)
# ============================================================
def build_encoder(seq_len, k_orders=(2, 4), q_depth=3):
    inputs = layers.Input(shape=(seq_len, 2))
    x = PhysicsBasisLayer(k_orders=k_orders, q_depth=q_depth)(inputs)
    x = layers.Conv1D(64,  3, padding="causal", dilation_rate=1, activation="relu")(x)
    x = layers.Conv1D(64,  3, padding="causal", dilation_rate=2, activation="relu")(x)
    x = layers.Conv1D(128, 3, padding="causal", dilation_rate=4, activation="relu")(x)
    x = layers.Conv1D(128, 3, padding="causal", dilation_rate=8, activation="relu")(x)
    z = layers.Conv1D(64, 1, padding="same")(x)
    return models.Model(inputs, z, name="tcn_physics_encoder")

def build_decoder(seq_len, channels=2, latent_dim=64):
    inputs = layers.Input(shape=(seq_len, latent_dim))
    x = layers.Conv1D(128, 3, padding="causal", dilation_rate=1, activation="relu")(inputs)
    x = layers.Conv1D(64,  3, padding="causal", dilation_rate=2, activation="relu")(x)
    x = layers.Conv1D(64,  3, padding="causal", dilation_rate=4, activation="relu")(x)
    outputs = layers.Conv1D(channels, 1, padding="same")(x)
    return models.Model(inputs, outputs, name="tcn_decoder")


# ============================================================
# 하이퍼파라미터
# ============================================================
N          = 64
seq_len    = N
back_off   = PA_IBO_DB  # IBO [dB] (포화 기준, dvae_curriculum.py에서 정의)
batch_size = _mod.EXP_BATCH_SIZE      # dvae_curriculum.py의 실험 설정
K_ORDERS   = (2, 4)     # 3, 5차 기저
Q_DEPTH    = 3          # q = 0, 1, 2
REF_POWER  = 1.0        # PA 기준 전력 (정규화된 원신호의 평균 전력)
SAVE_INTERVAL     = _mod.EXP_SAVE_INTERVAL
USE_CURRICULUM    = _mod.EXP_USE_CURRICULUM
DATA_FRACTION     = _mod.EXP_DATA_FRACTION
CURRICULUM_STAGES = _mod.exp_stages()

# DLA 손실 (dvae_curriculum.dla_loss)
#   'fixed'     : mean|G_FIX·x - y|^2  (목표 이득 G_FIX는 PA 차수에 따라 dvae_curriculum.py에서 정함)
#   'gain_free' : 분석용 (같은 출력 이득에서는 fixed와 최적점이 같고 학습이 불안정 — 권장하지 않음)
LOSS_MODE    = 'fixed'
G_TARGET     = 0.975
GAIN_PENALTY = 100.0
LOSS_TAG     = '' if LOSS_MODE == 'fixed' else '_gainfree'

CKPT_PREFIX = _mod.ckpt_prefix('vtae') + LOSS_TAG
HISTORY_CSV = os.path.join(MODEL_DIR, f'training_history_{CKPT_PREFIX}.csv')
print(f"체크포인트 접두어: {CKPT_PREFIX}, 배치 {batch_size}, 데이터 비율 {DATA_FRACTION}, 커리큘럼 {USE_CURRICULUM}")


def find_latest_checkpoint(model_dir, prefix):
    pattern = re.compile(re.escape(prefix) + r'_stage(\d+)_epoch(\d+)\.weights\.h5')
    cands = []
    for fname in os.listdir(model_dir):
        m = pattern.fullmatch(fname)
        if m:
            cands.append((int(m.group(1)), int(m.group(2)), os.path.join(model_dir, fname)))
    if not cands:
        return 0, 0, None
    cands.sort(key=lambda t: (t[0], t[1]))
    return cands[-1]


# ============================================================
# 데이터 로드 (원신호만 사용, 물리적 시간 순서의 스트림)
# ============================================================
outputData_I = pd.read_csv(os.path.join(BASE_DIR, 'input_target_I.csv'), header=None).to_numpy()
outputData_Q = pd.read_csv(os.path.join(BASE_DIR, 'input_target_Q.csv'), header=None).to_numpy()
if outputData_I.shape[0] % seq_len != 0:
    raise ValueError(f"샘플 수({outputData_I.shape[0]})가 seq_len({seq_len})의 배수가 아닙니다.")

# 연속된 64샘플씩 시퀀스로 자름
outputData_I = outputData_I.reshape(-1, seq_len)
outputData_Q = outputData_Q.reshape(-1, seq_len)
if DATA_FRACTION < 1.0:                     # 학습 예산 제한: 앞부분 연속 구간만 사용
    n_use = int(DATA_FRACTION * outputData_I.shape[0])
    outputData_I, outputData_Q = outputData_I[:n_use], outputData_Q[:n_use]
len_i   = outputData_I.shape[0]
n_train = int(0.8 * len_i)
n_val   = int(0.1 * len_i)
print(f"총 시퀀스 수: {len_i} (train {n_train}, val {n_val}, test {len_i - n_train - n_val})")

_, _, rms_train = normalize_with_rms(outputData_I[:n_train], outputData_Q[:n_train])
print(f"rms_train: {rms_train:.6f}")

def split(a, b):
    return to_1ch(a / rms_train, b / rms_train)

X_in_train_ch = split(outputData_I[:n_train],               outputData_Q[:n_train])
X_in_val_ch   = split(outputData_I[n_train:n_train + n_val], outputData_Q[n_train:n_train + n_val])
X_in_test_ch  = split(outputData_I[n_train + n_val:],        outputData_Q[n_train + n_val:])
print(f"정규화된 학습 신호 평균 전력: {np.mean(np.sum(X_in_train_ch**2, axis=-1)):.6f} (1이어야 함)")


# ============================================================
# 모델 생성
# ============================================================
dvae = DVAE_Curriculum(
    encoder      = build_encoder(seq_len, k_orders=K_ORDERS, q_depth=Q_DEPTH),
    decoder      = build_decoder(seq_len),
    backoff      = float(back_off),
    memory_depth = CURRICULUM_STAGES[0]["memory_depth"],
    snr_db       = CURRICULUM_STAGES[0]["snr_db"],
    ref_power    = REF_POWER,
    loss_mode    = LOSS_MODE,
    g_target     = G_TARGET,
    gain_penalty = GAIN_PENALTY,
)
print(f"DLA 손실: {LOSS_MODE}" + (f" (G_TARGET {G_TARGET}, 벌점 {GAIN_PENALTY})" if LOSS_MODE != 'fixed' else "")
      + f", 체크포인트 접두어 '{CKPT_PREFIX}'")
dvae.compile(optimizer=tf.keras.optimizers.Adam(CURRICULUM_STAGES[0]["init_lr"], clipnorm=1.0))
dvae.build((None, seq_len, 2))
print(f"총 파라미터 수: {dvae.encoder.count_params() + dvae.decoder.count_params():,}")

# 기울기 흐름 확인
sample_x = tf.constant(X_in_train_ch[:128])
with tf.GradientTape() as tape:
    pred = dvae(sample_x, training=True)
    loss_check, _ = dla_loss(sample_x, polynomial_tf(
        pred, backoff=float(back_off), memory_depth=CURRICULUM_STAGES[0]["memory_depth"],
        ref_power=REF_POWER), LOSS_MODE, G_TARGET, GAIN_PENALTY)
grad_norms = [tf.norm(g).numpy() for g in tape.gradient(loss_check, dvae.trainable_variables)
              if g is not None]
print(f"gradient 최소 {min(grad_norms):.2e}, 최대 {max(grad_norms):.2e}, "
      f"0인 레이어 {sum(1 for g in grad_norms if g < 1e-10)}개\n")


# ============================================================
# 콜백
# ============================================================
def evm_percent(x, y):
    return float(tf.sqrt(tf.reduce_mean(tf.square(x - y)) / tf.reduce_mean(tf.square(x))) * 100.0)

def evm_gain_norm(x, y):
    """이득 정규화 EVM [%]과 |g| (평가 파일 evm_pct와 같은 정의, 시간 영역)"""
    xc = tf.complex(x[..., 0], x[..., 1]); yc = tf.complex(y[..., 0], y[..., 1])
    g  = tf.reduce_sum(tf.math.conj(xc) * yc) / tf.cast(tf.reduce_sum(tf.abs(xc) ** 2), tf.complex64)
    e  = tf.reduce_mean(tf.abs(yc / g - xc) ** 2) / tf.reduce_mean(tf.abs(xc) ** 2)
    return float(tf.sqrt(e) * 100.0), float(tf.abs(g))

class EVMHistory(tf.keras.callbacks.Callback):
    """매 epoch 검증 데이터로 전체 모델(Q = 2) 기준 EVM을 계산하고 CSV에 기록"""
    def __init__(self, val_x, backoff, ref_power, csv_path):
        super().__init__()
        self.val_x, self.backoff, self.ref_power, self.csv_path = \
            tf.constant(val_x), backoff, ref_power, csv_path
        self.stage = 0
        self.global_epoch = 0
        if os.path.exists(csv_path):                       # 재개 시 이어서 기록
            self.global_epoch = max(0, sum(1 for _ in open(csv_path)) - 1)
        else:
            with open(csv_path, 'w', newline='') as f:
                csv.writer(f).writerow(['global_epoch', 'stage', 'memory_depth', 'loss', 'val_loss',
                                        'lr', 'evm_dpd_full_pct', 'evm_hpa_full_pct',
                                        'evm_dpd_gn_pct', 'evm_hpa_gn_pct', 'gain_dpd', 'gain_hpa'])
        hpa0 = polynomial_tf(self.val_x, backoff=backoff, memory_depth=2, ref_power=ref_power)
        self.evm_hpa = evm_percent(self.val_x, hpa0)
        self.evm_hpa_gn, self.g_hpa = evm_gain_norm(self.val_x, hpa0)

    def on_epoch_end(self, epoch, logs=None):
        self.global_epoch += 1
        pred = self.model(self.val_x, training=False)
        y    = polynomial_tf(pred, backoff=self.backoff, memory_depth=2, ref_power=self.ref_power)
        evm  = evm_percent(self.val_x, y)
        evm_gn, g = evm_gain_norm(self.val_x, y)
        lr   = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        loss, vloss = logs.get('loss', np.nan), logs.get('val_loss', np.nan)
        with open(self.csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([self.global_epoch, self.stage + 1, self.model._memory_depth,
                                    f'{loss:.6e}', f'{vloss:.6e}', f'{lr:.3e}',
                                    f'{evm:.4f}', f'{self.evm_hpa:.4f}',
                                    f'{evm_gn:.4f}', f'{self.evm_hpa_gn:.4f}', f'{g:.4f}', f'{self.g_hpa:.4f}'])
        print(f"\n[Stage {self.stage + 1}, epoch {epoch + 1}] EVM(Q=2, 이득 정규화) DPD {evm_gn:.2f}% "
              f"(|g| {g:.4f}) vs HPA {self.evm_hpa_gn:.2f}% (|g| {self.g_hpa:.4f}) | 비정규화 DPD {evm:.2f}% "
              f"| loss {loss:.2e} | val_loss {vloss:.2e} | lr {lr:.2e}")

class PeriodicSaver(tf.keras.callbacks.Callback):
    """단계 안에서 SAVE_INTERVAL epoch마다 가중치 저장"""
    def __init__(self, model_dir, prefix, interval):
        super().__init__()
        self.model_dir, self.prefix, self.interval, self.stage = model_dir, prefix, interval, 0
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0:
            path = os.path.join(self.model_dir, f"{self.prefix}_stage{self.stage}_epoch{epoch + 1}.weights.h5")
            self.model.save_weights(path)
            print(f"저장: {path}")

history_cb = EVMHistory(X_in_val_ch[:512], float(back_off), REF_POWER, HISTORY_CSV)
saver_cb   = PeriodicSaver(MODEL_DIR, CKPT_PREFIX, SAVE_INTERVAL)


# ============================================================
# 재개
# ============================================================
scheduler = CurriculumScheduler()
resume_stage, resume_epoch, resume_path = find_latest_checkpoint(MODEL_DIR, CKPT_PREFIX)
if resume_path is not None:
    print(f"저장된 가중치 발견: {resume_path} → 이어서 학습")
    dvae.load_weights(resume_path)
    if USE_CURRICULUM:
        for _ in range(resume_stage):
            scheduler.advance()
else:
    print("저장된 가중치 없음 — 처음부터 학습합니다.")


# ============================================================
# 학습 루프 (단계마다 fit 한 번)
# ============================================================
for stage_idx in range(resume_stage, len(CURRICULUM_STAGES)):
    cfg = CURRICULUM_STAGES[stage_idx]
    epoch_start = resume_epoch if stage_idx == resume_stage else 0
    if epoch_start >= cfg["total_epochs"]:
        print(f"Stage {stage_idx + 1}는 이미 완료 — 다음 단계로")
        if USE_CURRICULUM:
            scheduler.advance()
        continue

    dvae.set_curriculum(memory_depth=cfg["memory_depth"], snr_db=cfg["snr_db"])
    # 중요: Keras는 첫 fit에서 train/test 함수를 tf.function으로 컴파일해 재사용한다.
    # memory_depth는 파이썬 값이라 컴파일 시점의 값(Q = 0)이 그래프에 고정되므로,
    # 단계를 바꿀 때마다 함수를 버려 새 Q로 다시 컴파일되게 한다.
    # (이 두 줄이 없으면 2·3단계도 실제로는 Q = 0 PA로 학습된다)
    dvae.train_function = None
    dvae.test_function = None
    dvae.optimizer.learning_rate = cfg["init_lr"]            # 단계 시작 시 학습률 초기화
    history_cb.stage = saver_cb.stage = stage_idx
    print(f"\n[Stage {stage_idx + 1}] Q = {cfg['memory_depth']}, epoch {epoch_start + 1}~"
          f"{cfg['total_epochs']}, 초기 학습률 {cfg['init_lr']:.2e}")

    lr_scheduler   = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                          patience=5, min_lr=1e-5, verbose=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                                      restore_best_weights=True, verbose=1)
    dvae.fit(X_in_train_ch, X_in_train_ch,
             validation_data=(X_in_val_ch, X_in_val_ch),
             initial_epoch=epoch_start, epochs=cfg["total_epochs"], batch_size=batch_size,
             callbacks=[history_cb, saver_cb, lr_scheduler, early_stopping], verbose=1)

    # 단계 종료 시점의 가중치(조기 종료 시 최적 epoch로 복원된 값)를 단계 마지막 번호로 저장
    final_path = os.path.join(MODEL_DIR, f"{CKPT_PREFIX}_stage{stage_idx}_epoch{cfg['total_epochs']}.weights.h5")
    dvae.save_weights(final_path)
    print(f"단계 완료, 저장: {final_path}")
    print("Test set 평가:", dvae.evaluate(X_in_test_ch, X_in_test_ch, verbose=0))
    if USE_CURRICULUM:
        scheduler.advance()

print("\n학습 완료. 학습 곡선 기록:", HISTORY_CSV)
