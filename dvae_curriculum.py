"""
dvae_curriculum.py  — 실험 설정(PA 차수, IBO, 목표 이득, 배치, 데이터 비율, 커리큘럼)도 이 파일에서 관리
- 메모리 다항식 HPA 모델 (표준 memory polynomial, numpy / TensorFlow 두 버전)
- PhysicsBasisLayer (Volterra 기저 층)
- CurriculumScheduler, DVAE_Curriculum (VT-AE, DLA 학습)

[변경 사항]
1) 표준 memory polynomial:  y[n] = sum_k sum_q c_kq * x[n-q] * |x[n-q]|^(k-1)
   (이전: 메모리 항에 지연된 '진폭'만 쓰고 현재 샘플의 위상을 곱함)
2) 고정 기준 전력: PA의 정규화 스케일을 입력 신호마다 다시 계산하지 않고,
   원신호의 평균 전력 ref_power로 고정한다.
3) 계수: Ding et al. (IEEE Trans. Commun., 2004)의 Class-AB PA 계수를 그대로 사용하고,
   소신호 이득이 1이 되도록 c10으로 나누어 정규화한다 (비선형 특성의 상대적 모양은 그대로).
4) IBO는 표준 정의대로 '포화 입력 전력 대비 평균 입력 전력'으로 정의한다.
   원 계수의 메모리 없는 AM/AM은 입력 진폭 A_SAT(≈0.758)에서 가장 평평해지고(포화),
   그 이상에서는 다항식이 물리적으로 맞지 않게 다시 급증하므로,
   PA 입력 진폭을 A_SAT로 제한(진폭 제한기)한 뒤 다항식을 적용한다.
      s = sqrt(ref_power * 10^(IBO/10)) / A_SAT,  u = clip(x / s, A_SAT),  y = F(u) * s
   → 평균 입력 전력이 포화 입력 전력보다 IBO dB 낮다.
5) DLA 손실 선택 (dla_loss, VT-AE와 Transformer 공용)
   'fixed'     : loss = mean|G_FIX·x - y|^2                 (목표 이득 G_FIX, 5차 PA에서는 1)
   'gain_free' : loss = mean|x - y/g|^2 + λ·(|g| - g_target)^2
                 g = Σ x* y / Σ |x|^2 (배치 전체의 최소제곱 복소 이득)
                 평가 지표(이득 정규화 EVM)와 같은 기준으로 학습한다.
                 y/g 항은 이득에 무관하므로 벌점이 없으면 이득이 떠돌고(학습 불안정),
                 이득을 낮추면 EVM이 계속 좋아진다(출력 전력을 줄여 back-off를 늘리는 것과 같음).
                 그래서 |g|를 g_target에 양방향 벌점으로 고정한다.
                 주의: 같은 출력 이득에서는 'fixed' 손실과 최적점이 사실상 같다
                 (fixed = |1-g|^2 + |d|^2, gain_free = |d|^2/|g|^2, d는 x와 무상관인 왜곡).
                 즉 이 손실은 성능 향상이라기보다 '출력 이득(전력)을 지정해 학습'하는 도구다.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


# =======================================================================
# 실험 설정 — 데이터 생성·학습·평가 파일이 모두 이 값을 읽는다 (여기만 바꾸면 됨)
# =======================================================================
PA_ORDER  = 5       # PA 모델: 5 = Ding et al. 측정 Class-AB PA (5차),
                    #          7, 9 = 고차 항을 가진 합성 PA (메모리 항은 Ding 계수 그대로)
PA_IBO_DB = 8.0     # IBO [dB] (포화 입력 전력 기준)

# DLA 목표 이득 G: loss = mean|G·x − PA(z)|^2
#   7·9차 PA는 포화 전부터 압축이 커서 이득 1을 목표로 하면 학습이 실패한다(DPD가 HPA보다 나빠짐).
#   그래서 HPA 자체 이득(5차 0.924, 7차 0.902, 9차 0.889)보다 약간 높은 값을 목표로 둔다.
G_TARGET_BY_ORDER = {5: 1.0, 7: 0.93, 9: 0.92}

# 학습 설정 (VT-AE, Transformer 공통 — 공정 비교를 위해 두 모델에 같은 값 사용)
EXP_BATCH_SIZE     = 512     # 배치 (64샘플 시퀀스 개수)
EXP_DATA_FRACTION  = 0.02    # 학습 데이터 중 사용할 비율 (앞에서부터 연속 구간)
                             # 0.02 → 약 8,000 시퀀스, 배치 512에서 epoch당 약 12회 갱신 (학습 예산 제한 조건)
                             # 1.0  → 전체 데이터
EXP_USE_CURRICULUM = False   # 논문 조건: 커리큘럼 없음
EXP_STAGES_CURRICULUM = {    # 단계: (memory depth Q, epoch 수, 초기 학습률)
    0: (0, 10, 1e-3),
    1: (1, 10, 5e-4),
    2: (2, 20, 2.5e-4),
}
EXP_STAGES_NOCURRICULUM = {0: (2, 100, 1e-3)}   # 논문 조건: Q = 2, 100 epoch, 학습률 1e-3
EXP_SAVE_INTERVAL = 10       # 체크포인트 저장 간격 (단계 안의 epoch 기준)

def exp_stages():
    st = EXP_STAGES_CURRICULUM if EXP_USE_CURRICULUM else EXP_STAGES_NOCURRICULUM
    return {k: {"memory_depth": q, "snr_db": -1.0, "total_epochs": e, "init_lr": lr}
            for k, (q, e, lr) in st.items()}

# 반복 학습 (초기 가중치에 따른 편차 확인)
#   EXP_SEED = None : seed를 고정하지 않음. 체크포인트 이름에 seed가 붙지 않음 (기존 학습 결과 = 1회차)
#   EXP_SEED = 2, 3, ... : tf.keras.utils.set_random_seed(seed)로 가중치 초기값·미니배치 순서·dropout을 고정하고
#                          체크포인트 이름 끝에 _s2, _s3 ...를 붙인다 (기존 가중치를 덮어쓰지 않음)
#   Colab에서 파일을 고치지 않고 바꾸려면 os.environ['EXP_SEED'] = '3' 처럼 환경변수로 지정해도 된다.
EXP_SEED = None
if os.environ.get('EXP_SEED'):
    EXP_SEED = int(os.environ['EXP_SEED'])
EXP_SEEDS_EVAL = [None, 2, 3, 4, 5]   # 평가 파일에서 평균·표준편차를 낼 seed 목록 (None = 기존 학습)

def seed_suffix():
    return '' if EXP_SEED is None else f'_s{EXP_SEED}'

def base_tag():
    """seed를 뺀 실험 이름 (같은 조건의 반복 학습들이 공유)"""
    return (f"pa{PA_ORDER}_ibo{PA_IBO_DB:g}_b{EXP_BATCH_SIZE}_f{EXP_DATA_FRACTION:g}_"
            f"{'cur' if EXP_USE_CURRICULUM else 'nocur'}")

def exp_tag():
    return base_tag() + seed_suffix()

# 일반적인 Transformer DPD (문헌 기준선, Transformer_DPD_conventional.py)
#   물리 기저 없음, 잔차 출력 없음, 커리큘럼 없음.  학습 방식은 아래에서 선택
EXP_CONV_TRAIN_MODE = 'dla'      # 'dla': 직접 학습 (VT-AE와 같은 손실 → 물리 기저·잔차 출력의 효과를 분리해 비교)
                                 # 'ila': 간접 학습 (PA 출력 → 입력 역함수를 학습해 앞에 복사, 반복 갱신)
                                 #   ※ 5차 PA, IBO 8 dB 시험에서 ILA는 반복할수록 발산해 HPA보다 나빠졌다
                                 #     (포화된 출력에서는 후치 역함수가 정의되지 않음). 사용 시 결과 해석 주의
EXP_CONV_INPUT      = 'iq_amp'   # 'iq': I, Q  /  'iq_amp': I, Q, |x|
EXP_CONV_ILA_ITERS  = 4          # ILA 반복 횟수 (전체 epoch를 나눠 씀)

def ckpt_prefix(model_type):
    """체크포인트 파일 접두어 — 설정이 다르면 이름이 달라져 서로 섞이지 않는다"""
    base = {'vtae': 'vtae', 'transformer': 'transformer', 'transformer_conv': 'transformer_conv'}[model_type]
    tag = base_tag()
    if model_type == 'transformer_conv':          # 커리큘럼을 쓰지 않으므로 학습 방식·입력으로 구분
        tag = tag.rsplit('_', 1)[0] + f"_{EXP_CONV_TRAIN_MODE}_{EXP_CONV_INPUT}"
    return base + '_' + tag + seed_suffix()           # seed는 맨 끝 (_s2 ...)

def total_epochs():
    return sum(cfg["total_epochs"] for cfg in exp_stages().values())

def stage_epoch_offsets():
    """단계별 누적 epoch 시작점 (체크포인트를 전체 epoch 번호로 바꿀 때 사용)"""
    off, acc = {}, 0
    for k, cfg in sorted(exp_stages().items()):
        off[k] = acc; acc += cfg["total_epochs"]
    return off


# -----------------------------------------------------------------------
# 메모리 다항식 계수
# 키 (k_idx, q): k_idx = 0, 2, 4, 6, 8 은 각각 1, 3, 5, 7, 9차 항,  q = 메모리 지연
#   항의 형태:  c * x[n-q] * |x[n-q]|^k_idx
# -----------------------------------------------------------------------
# 5차: Ding et al. (IEEE Trans. Commun., 2004)의 Class-AB PA 계수
DING_C_VALS = {
    (0, 0):  1.0513 + 0.0904j,     # c10
    (2, 0): -0.0542 - 0.2900j,     # c30
    (4, 0): -0.9657 - 0.7028j,     # c50
    (0, 1): -0.0680 - 0.0023j,     # c11
    (2, 1):  0.2234 + 0.2317j,     # c31
    (4, 1): -0.2451 - 0.3735j,     # c51
    (0, 2):  0.0289 - 0.0054j,     # c12
    (2, 2): -0.0621 - 0.0932j,     # c32
    (4, 2):  0.1229 + 0.1508j,     # c52
}
_DING_MEMORY = {k: v for k, v in DING_C_VALS.items() if k[1] > 0}

# 7·9차 합성 PA: 메모리 없는 항(q = 0)은 Rapp 모델(p = 2, 포화 0.75) AM/AM과
#   최대 20°의 Saleh형 AM/PM 곡선 a·exp(j·20°·a²/(1+a²))에 홀수차 다항식을 최소제곱으로 맞춘 값,
#   c10은 Ding과 같게 맞추고, 메모리 항(q = 1, 2)은 Ding 계수를 그대로 사용.
PA7_C_VALS = {
    (0, 0):  1.051300 + 0.090400j,
    (2, 0): -0.243213 + 0.316162j,
    (4, 0): -0.285862 - 0.340760j,
    (6, 0):  0.177535 + 0.118536j,
    **_DING_MEMORY,
}
PA9_C_VALS = {
    (0, 0):  1.051300 + 0.090400j,
    (2, 0): -0.059480 + 0.375440j,
    (4, 0): -0.896071 - 0.524825j,
    (6, 0):  0.896248 + 0.331042j,
    (8, 0): -0.279518 - 0.082006j,
    **_DING_MEMORY,
}
PA_C_VALS = {5: DING_C_VALS, 7: PA7_C_VALS, 9: PA9_C_VALS}[PA_ORDER]
# 소신호 이득 1로 정규화 (모든 계수를 c10으로 나눔)
ALL_C_VALS = {k: v / PA_C_VALS[(0, 0)] for k, v in PA_C_VALS.items()}
G_FIX = G_TARGET_BY_ORDER[PA_ORDER]

def _saturation_amplitude(c_vals):
    """포화 입력 진폭: 메모리 없는 AM/AM |F0(a)|이 처음 최대가 되는 점,
    최대가 없으면 기울기가 가장 작은(가장 평평한) 점 (Ding 5차 ≈ 0.758, 7차 ≈ 0.983, 9차 ≈ 1.111)"""
    a = np.linspace(1e-4, 1.3, 130000)
    f0 = sum(c * a ** (k + 1) for (k, q), c in c_vals.items() if q == 0)
    g = np.gradient(np.abs(f0), a)
    return float(a[np.argmax(g <= 0)]) if np.any(g <= 0) else float(a[np.argmin(g)])

A_SAT = _saturation_amplitude(ALL_C_VALS)


# -----------------------------------------------------------------------
# numpy 버전 HPA (데이터 생성, 평가에서 사용)
# -----------------------------------------------------------------------
def polynomial_np(x, backoff, ref_power=None, memory_depth=2):
    """
    x            : 복소 기저대역 신호 (물리적 시간 순서), 임의 shape
    backoff      : IBO [dB] — 평균 입력 전력이 포화 입력 전력보다 backoff dB 낮음
    ref_power    : PA 기준 전력 (원신호의 평균 전력). None이면 x 자신의 평균 전력
                   → DPD 출력을 넣을 때는 반드시 원신호의 평균 전력을 넘겨야 PA가 고정된다.
    memory_depth : 사용할 최대 메모리 지연 q
    """
    shape = x.shape
    u = np.asarray(x, dtype=complex).reshape(-1)
    if ref_power is None:
        ref_power = np.mean(np.abs(u) ** 2)
    s = np.sqrt(ref_power * 10 ** (backoff / 10)) / A_SAT
    u = u / s
    mag = np.abs(u)
    u = np.where(mag > A_SAT, u / np.maximum(mag, 1e-12) * A_SAT, u)   # 진폭 제한기 (포화)
    y = np.zeros_like(u)
    for (k, q), c in ALL_C_VALS.items():
        if q > memory_depth:
            continue
        ud = u if q == 0 else np.concatenate([np.zeros(q, dtype=complex), u[:-q]])
        y += c * ud * np.abs(ud) ** k
    return (y * s).reshape(shape)


# -----------------------------------------------------------------------
# TensorFlow 버전 HPA (DLA 학습에서 사용, 실수 연산만 사용해 기울기 안정)
# -----------------------------------------------------------------------
def polynomial_tf(p_in_2ch, backoff=PA_IBO_DB, snr_db=-1.0, memory_depth=2, ref_power=1.0):
    """
    p_in_2ch  : (batch, seq_len, 2) — [I, Q]
    ref_power : PA 기준 전력. 학습 데이터는 원신호 RMS로 정규화되어 평균 전력이 1이므로 기본값 1.0
    반환       : (batch, seq_len, 2) — HPA 출력
    시퀀스 시작 부분의 지연 샘플은 0으로 채운다.
    """
    s = float(np.sqrt(ref_power * 10.0 ** (backoff / 10.0)) / A_SAT)
    I = p_in_2ch[..., 0] / s
    Q = p_in_2ch[..., 1] / s
    mag   = tf.sqrt(I ** 2 + Q ** 2 + 1e-12)
    limit = tf.minimum(1.0, A_SAT / mag)                # 진폭 제한기 (포화)
    I, Q  = I * limit, Q * limit

    yr = tf.zeros_like(I)
    yi = tf.zeros_like(I)
    for (k, q), c in ALL_C_VALS.items():
        if q > memory_depth:
            continue
        if q == 0:
            Id, Qd = I, Q
        else:
            Id = tf.pad(I[:, :-q], [[0, 0], [q, 0]])
            Qd = tf.pad(Q[:, :-q], [[0, 0], [q, 0]])
        if k == 0:
            g = 1.0
        else:
            g = tf.pow(Id ** 2 + Qd ** 2, k // 2)       # |x|^k (k는 짝수 → 제곱근 불필요)
        cr, ci = float(np.real(c)), float(np.imag(c))
        yr = yr + (cr * Id - ci * Qd) * g                # 복소 곱셈 c * x_d 의 실수부
        yi = yi + (cr * Qd + ci * Id) * g                #                    허수부

    hpa_real = yr * s
    hpa_imag = yi * s

    if snr_db >= 0.0:
        signal_power = tf.reduce_mean(hpa_real ** 2 + hpa_imag ** 2)
        snr_linear   = tf.constant(10.0 ** (snr_db / 10.0), dtype=tf.float32)
        noise_std    = tf.sqrt(signal_power / snr_linear / 2.0)
        hpa_real    += tf.random.normal(tf.shape(hpa_real), stddev=noise_std)
        hpa_imag    += tf.random.normal(tf.shape(hpa_imag), stddev=noise_std)

    return tf.stack([hpa_real, hpa_imag], axis=-1)


# -----------------------------------------------------------------------
# PhysicsBasisLayer — Volterra(메모리 다항식) 기저 생성 층
# 입력 (batch, seq_len, 2) → 출력 (batch, seq_len, 2 + 2*K*Q)
# 기저 항: x[n-q] * |x[n-q]|^k  (k = 2, 4 → 3, 5차 항),  q = 0, ..., q_depth-1
# -----------------------------------------------------------------------
@tf.keras.utils.register_keras_serializable()
class PhysicsBasisLayer(layers.Layer):
    def __init__(self, k_orders=(2, 4), q_depth=3, **kwargs):
        super().__init__(**kwargs)
        self.k_orders = tuple(k_orders)
        self.q_depth  = q_depth

    def call(self, inputs):
        I = inputs[..., 0]
        Q = inputs[..., 1]
        feats = [I, Q]
        for q in range(self.q_depth):
            if q == 0:
                I_d, Q_d = I, Q
            else:
                I_d = tf.pad(I[:, :-q], [[0, 0], [q, 0]])
                Q_d = tf.pad(Q[:, :-q], [[0, 0], [q, 0]])
            mag2 = I_d ** 2 + Q_d ** 2
            for k in self.k_orders:
                gain = tf.pow(mag2, k // 2)               # |x|^k
                feats.append(I_d * gain)
                feats.append(Q_d * gain)
        return tf.stack(feats, axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 2 + 2 * len(self.k_orders) * self.q_depth)

    def get_config(self):
        config = super().get_config()
        config.update({"k_orders": list(self.k_orders), "q_depth": self.q_depth})
        return config


# -----------------------------------------------------------------------
# CurriculumScheduler — 메모리 깊이를 Q = 0 → 1 → 2로 늘림 (학습 중 노이즈 없음)
# -----------------------------------------------------------------------
class CurriculumScheduler:
    STAGES = [
        {"memory_depth": 0, "snr_db": -1.0, "description": "Stage 1 | memory depth Q = 0"},
        {"memory_depth": 1, "snr_db": -1.0, "description": "Stage 2 | memory depth Q = 1"},
        {"memory_depth": 2, "snr_db": -1.0, "description": "Stage 3 | memory depth Q = 2"},
    ]

    def __init__(self):
        self._stage = 0

    @property
    def stage(self):
        return self._stage

    @property
    def memory_depth(self):
        return self.STAGES[self._stage]["memory_depth"]

    @property
    def snr_db(self):
        return self.STAGES[self._stage]["snr_db"]

    @property
    def description(self):
        return self.STAGES[self._stage]["description"]

    @property
    def is_final_stage(self):
        return self._stage == len(self.STAGES) - 1

    def advance(self):
        if not self.is_final_stage:
            self._stage += 1
            print(f"\n{'='*50}\n커리큘럼 단계 전환 → {self.description}\n{'='*50}\n")

    def status(self):
        print(f"현재 커리큘럼: {self.description}")


# -----------------------------------------------------------------------
# DLA 손실 (VT-AE, Transformer 공용)
# -----------------------------------------------------------------------
LOSS_MODES = ('fixed', 'gain_free')

def dla_loss(x, y, mode='fixed', g_target=0.975, gain_penalty=100.0):
    """
    x : 원신호 (batch, seq_len, 2),  y : HPA 출력 (batch, seq_len, 2)
    반환: (loss, |g|)   |g|는 로그 확인용 (fixed 모드에서도 계산)
    """
    xr, xi = x[..., 0], x[..., 1]
    yr, yi = y[..., 0], y[..., 1]
    den = tf.reduce_sum(xr ** 2 + xi ** 2) + 1e-12
    gr  = tf.reduce_sum(xr * yr + xi * yi) / den          # Re{Σ x* y} / Σ|x|^2
    gi  = tf.reduce_sum(xr * yi - xi * yr) / den          # Im{Σ x* y} / Σ|x|^2
    g2  = gr ** 2 + gi ** 2 + 1e-12
    g_abs = tf.sqrt(g2)
    if mode == 'fixed':
        return tf.reduce_mean(tf.square(G_FIX * x - y)), g_abs
    if mode != 'gain_free':
        raise ValueError(f"loss mode는 {LOSS_MODES} 중 하나: {mode}")
    zr = (yr * gr + yi * gi) / g2                          # y / g = y g* / |g|^2
    zi = (yi * gr - yr * gi) / g2
    mse = tf.reduce_mean((xr - zr) ** 2 + (xi - zi) ** 2) / 2.0   # fixed 모드와 같은 크기(채널 평균)
    penalty = gain_penalty * tf.square(g_abs - g_target)        # 이득을 목표값에 고정 (양방향)
    return mse + penalty, g_abs


# -----------------------------------------------------------------------
# DVAE_Curriculum (VT-AE) — DLA 학습:  loss = dla_loss(x, HPA(x + Δ))
# -----------------------------------------------------------------------
@tf.keras.utils.register_keras_serializable()
class DVAE_Curriculum(models.Model):

    def __init__(self, encoder, decoder, backoff=PA_IBO_DB,
                 memory_depth=0, snr_db=-1.0, ref_power=1.0,
                 loss_mode='fixed', g_target=0.975, gain_penalty=100.0, **kwargs):
        super().__init__(**kwargs)
        if loss_mode not in LOSS_MODES:
            raise ValueError(f"loss_mode는 {LOSS_MODES} 중 하나: {loss_mode}")
        self.loss_mode     = loss_mode
        self.g_target         = g_target
        self.gain_penalty  = gain_penalty
        self.encoder       = encoder
        self.decoder       = decoder
        self.backoff       = backoff
        self.ref_power     = ref_power      # PA 기준 전력 (학습 데이터는 1.0)
        self._memory_depth = memory_depth
        self._snr_db       = snr_db

    def set_curriculum(self, memory_depth: int, snr_db: float):
        self._memory_depth = memory_depth
        self._snr_db       = snr_db

    def get_curriculum(self):
        return {"memory_depth": self._memory_depth, "snr_db": self._snr_db}

    def build(self, input_shape):
        if not self.encoder.built:
            self.encoder.build(input_shape)
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            "encoder_config": self.encoder.get_config(),
            "decoder_config": self.decoder.get_config(),
            "backoff":        self.backoff,
            "ref_power":      self.ref_power,
            "memory_depth":   self._memory_depth,
            "snr_db":         self._snr_db,
            "loss_mode":      self.loss_mode,
            "g_target":          self.g_target,
            "gain_penalty":   self.gain_penalty,
        })
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        backoff      = config.pop("backoff", PA_IBO_DB)
        ref_power    = config.pop("ref_power", 1.0)
        memory_depth = config.pop("memory_depth", 0)
        snr_db       = config.pop("snr_db", -1.0)
        loss_mode    = config.pop("loss_mode", 'fixed')
        g_target        = config.pop("g_target", 0.975)
        gain_penalty = config.pop("gain_penalty", 100.0)
        enc_cfg      = config.pop("encoder_config")
        dec_cfg      = config.pop("decoder_config")
        custom_objects = dict(custom_objects or {})
        custom_objects['PhysicsBasisLayer'] = PhysicsBasisLayer
        encoder = tf.keras.models.Model.from_config(enc_cfg, custom_objects=custom_objects)
        decoder = tf.keras.models.Model.from_config(dec_cfg, custom_objects=custom_objects)
        return cls(encoder=encoder, decoder=decoder, backoff=backoff, ref_power=ref_power,
                   memory_depth=memory_depth, snr_db=snr_db, loss_mode=loss_mode,
                   g_target=g_target, gain_penalty=gain_penalty, **config)

    def compile(self, optimizer, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer
        self.loss_tracker = {"loss": tf.keras.metrics.Mean(name="loss"),
                             "gain": tf.keras.metrics.Mean(name="gain")}

    @property
    def metrics(self):
        return list(self.loss_tracker.values())

    def call(self, inputs, training=False):
        z          = self.encoder(inputs, training=training)
        correction = self.decoder(z, training=training)
        return inputs + correction                       # 전치왜곡 출력 (residual)

    def _hpa(self, predistorted):
        return polynomial_tf(predistorted, backoff=self.backoff, snr_db=self._snr_db,
                             memory_depth=self._memory_depth, ref_power=self.ref_power)

    def _loss(self, x, training):
        y = self._hpa(self(x, training=training))
        return dla_loss(x, y, self.loss_mode, self.g_target, self.gain_penalty)

    def train_step(self, data):
        x, _ = data
        with tf.GradientTape() as tape:
            loss, g_abs = self._loss(x, True)
        weights = self.encoder.trainable_weights + self.decoder.trainable_weights
        grads = tape.gradient(loss, weights)
        self.optimizer.apply_gradients(zip(grads, weights))
        self.loss_tracker["loss"].update_state(loss)
        self.loss_tracker["gain"].update_state(g_abs)
        return {k: v.result() for k, v in self.loss_tracker.items()}

    def test_step(self, data):
        x, _ = data
        loss, g_abs = self._loss(x, False)
        self.loss_tracker["loss"].update_state(loss)
        self.loss_tracker["gain"].update_state(g_abs)
        return {k: v.result() for k, v in self.loss_tracker.items()}


# -----------------------------------------------------------------------
# 일반적인 Transformer DPD (문헌 기준선) — 학습 파일과 평가 파일이 같은 구조를 쓰도록 여기서 정의
#   입력: I, Q (+ |x|) 시퀀스 → 선형 임베딩 → 위치 부호화 → Transformer 인코더 2개 → 선형 출력 (I, Q)
#   물리 기저 없음, 잔차 연결 없음 (전치왜곡 신호를 직접 출력)
# -----------------------------------------------------------------------
@tf.keras.utils.register_keras_serializable()
class AmplitudeFeature(layers.Layer):
    """(I, Q) → (I, Q, |x|)"""
    def call(self, inputs):
        amp = tf.sqrt(tf.reduce_sum(tf.square(inputs), axis=-1, keepdims=True) + 1e-12)
        return tf.concat([inputs, amp], axis=-1)
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (3,)

@tf.keras.utils.register_keras_serializable()
class SinusoidalPositionalEncoding(layers.Layer):
    def __init__(self, sequence_length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length, self.d_model = sequence_length, d_model
        pos = np.arange(sequence_length)[:, None]
        idx = np.arange(d_model)[None, :]
        ang = pos / np.power(10000.0, (2 * (idx // 2)) / d_model)
        pe = np.concatenate([np.sin(ang[:, 0::2]), np.cos(ang[:, 1::2])], -1)
        self.pe = tf.constant(pe[None, ...], dtype=tf.float32)
    def call(self, inputs):
        return inputs + self.pe[:, :tf.shape(inputs)[1], :]
    def get_config(self):
        c = super().get_config(); c.update({"sequence_length": self.sequence_length, "d_model": self.d_model})
        return c

def build_transformer_conventional(seq_len, input_mode=EXP_CONV_INPUT, d_model=64, num_heads=4,
                                   key_dim=64, ff_dim=128, num_layers=2, dropout=0.1):
    """파라미터 약 16.7만 개 (VT-AE 15.9만, 기존 Transformer 16.7만과 비슷하게 맞춤)"""
    inputs = layers.Input(shape=(seq_len, 2))
    x = AmplitudeFeature()(inputs) if input_mode == 'iq_amp' else inputs
    x = layers.Dense(d_model)(x)
    x = SinusoidalPositionalEncoding(seq_len, d_model)(x)
    for _ in range(num_layers):
        a = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
        a = layers.Dropout(dropout)(a)
        x = layers.LayerNormalization(epsilon=1e-6)(x + a)
        f = layers.Dense(ff_dim, activation="relu")(x)
        f = layers.Dense(d_model)(f)
        f = layers.Dropout(dropout)(f)
        x = layers.LayerNormalization(epsilon=1e-6)(x + f)
    outputs = layers.Dense(2)(x)                     # 잔차 없이 직접 출력
    return models.Model(inputs, outputs, name="Transformer_DPD_conventional")
