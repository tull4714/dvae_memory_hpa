"""
DPD 평가: EVM(노이즈 없음, AWGN), BER(AWGN, Rayleigh, Rician), SNR penalty — VT-AE와 Transformer 동시 평가

[AWGN BER 추가]
- ber_awgn(): 페이딩 없는 AWGN BER. 페이딩 BER과 다른 난수열(seed 24680)을 써서 기존 Rayleigh/Rician 값은 바뀌지 않는다.
- AWGN SNR penalty 표 (TARGET_BER_AWGN = 1e-3, 1e-4, 1e-5), .mat에 ber_awgn_* 저장, 반복 학습 통계에 AWGN penalty 추가

[이번 변경]
1) 실험 설정(PA 차수, IBO, 배치, 데이터 비율, 커리큘럼)은 dvae_curriculum.py에서 읽고,
   같은 설정으로 학습한 두 모델의 체크포인트를 자동으로 찾는다.
2) MODELS의 모델을 같은 잡음·페이딩 실현값으로 함께 평가한다.
2-1) 'transformer_conv': 일반적인 Transformer DPD(물리 기저·잔차·커리큘럼 없음, ILA 또는 DLA)도 함께 평가.
3) GAIN_CORRECT_BER = True: 수신 심볼을 각 방식의 대역 내 LS 이득 ĝ로 나눈 뒤 판정
   (실제 수신기는 채널 추정에 PA 이득이 포함됨). 이득이 1이 아닌 HPA/DPD의 BER이 공정해진다.
4) Rician SNR penalty 표를 출력한다.
5) EVAL_CURVE = True: 모든 체크포인트(10 epoch 간격)를 작은 신호로 평가해 epoch별 EVM·BER 곡선 저장.

[변경 사항]
1) HPA: dvae_curriculum.py의 polynomial_np (표준 memory polynomial) 사용.
   DPD 출력을 PA에 넣을 때도 PA 기준 전력은 원신호의 평균 전력으로 고정 (PA는 고정된 장치).
2) 시간 신호를 물리적 전송 순서로 펼치고(order='F'), 복조도 같은 순서로 되돌림.
3) 평가 신호는 학습 데이터와 다른 시드(SEED = 2)로 생성 → 학습에 쓴 신호로 평가하지 않음.
4) 노이즈 전력은 방식마다 자기 송신 신호의 평균 전력 기준 (같은 Eb/N0에서 비교),
   노이즈와 페이딩 실현값은 세 방식에 공통으로 사용.
5) BER 판정 함수 decision()을 벡터화 (기존 반복문과 결과 동일, 수백 배 빠름).
6) 결과를 MATLAB 그림용 .mat 파일로 저장.
"""
import os
import sys
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy.io import savemat
import importlib.util, shutil

# ============================================================
# 설정
# ============================================================
MODELS     = ['vtae', 'transformer_conv']   # 평가할 모델 (논문: VT-AE와 일반 Transformer) (같은 잡음·페이딩으로 함께 평가)
#   'vtae'             : 제안 VT-AE (IQ_Conv1D_DVAE_with_LSTM_v7.py)
#   'transformer'      : 같은 학습 체계(물리 기저, 잔차, DLA, 커리큘럼)의 Transformer — 백본만 다른 절제 비교
#   'transformer_conv' : 일반적인 Transformer DPD (Transformer_DPD_conventional.py) — 문헌 기준선
#   학습하지 않은 모델은 목록에서 빼면 된다.
BASE_DIR   = '/content/drive/MyDrive/NonlinearMemory'
MODEL_DIR  = os.path.join(BASE_DIR, 'vt-ae')
LOSS_TAG   = ''            # 학습 파일 LOSS_MODE가 'fixed'면 '' ('gain_free'면 '_gainfree')
GAIN_CORRECT_BER = True    # True: 수신기가 PA 이득까지 포함해 채널을 추정 (rx/(h·ĝ)) — 실제 수신기와 같음
                           # False: 이전 방식 (rx/h, 이득 오차가 그대로 BER 열화로 나타남)
SAVE_AMPM  = True          # AM/AM, AM/PM 그림(am_pm_plot.m)용 DPD 출력 저장
EVAL_CURVE = True          # True: 저장된 모든 체크포인트를 작은 신호로 평가해 epoch별 학습 곡선 출력·저장
CURVE_BLOCK = 4000         # 학습 곡선용 OFDM 심볼 수 (25.6만 심볼, 비트 102만 개)
CURVE_SNR   = np.arange(10, 20, 2)
TARGET_BER  = [1e-2, 3e-3, 1e-3]       # SNR penalty를 계산할 목표 BER (Rician)
TARGET_BER_AWGN = [1e-3, 1e-4, 1e-5]   # AWGN SNR penalty 목표 BER (AWGN은 BER 곡선이 가팔라 더 낮은 BER에서 비교)
EVAL_REPEAT = True         # True: dvae_curriculum.EXP_SEEDS_EVAL의 반복 학습들을 평가해 평균·표준편차·95% 신뢰구간 출력
REPEAT_BER  = True         # True: 반복 학습마다 Rician penalty도 계산 (전체 평가 신호, seed당 수 분)

Fs, Fd   = 4, 1
N        = 64
seq_len  = N
Block    = 100000
M        = 16
D        = N * Block
SEED     = 2               # 평가용 시드 (학습 데이터는 1)
SNR      = np.arange(0, 20, 2)          # Eb/N0 [dB]

# dvae_curriculum.py 로드
# dvae_curriculum.py를 파일 경로로 직접 로드 (패키지/캐시 문제 회피)
_dvae_path = os.path.join(BASE_DIR, 'dvae_curriculum.py')
_pycache   = os.path.join(BASE_DIR, '__pycache__')
_path = _dvae_path
if os.path.exists(_pycache):
    shutil.rmtree(_pycache)
for _key in list(sys.modules.keys()):
    if 'dvae_curriculum' in _key:
        del sys.modules[_key]
_spec = importlib.util.spec_from_file_location("dvae_curriculum", _path)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
DVAE_Curriculum   = _mod.DVAE_Curriculum
PhysicsBasisLayer = _mod.PhysicsBasisLayer
polynomial_np     = _mod.polynomial_np
back_off          = _mod.PA_IBO_DB    # IBO [dB] (포화 기준)
EXP_TAG           = _mod.exp_tag()
print(f"실험 설정: {EXP_TAG}, PA {_mod.PA_ORDER}차, A_SAT {_mod.A_SAT:.3f}, 목표 이득 {_mod.G_FIX}")


# ============================================================
# 신호 처리 함수
# ============================================================
def gen_mapping(M, D, seed):
    rng = np.random.RandomState(seed)
    rand_I = rng.rand(1, D)
    rand_Q = rng.rand(1, D)
    if M == 2:
        s = 2 * np.fix(rand_I * 2) - 1
    elif M == 4:
        s = (2 * np.fix(rand_I * 2) - 1) + 1j * (2 * np.fix(rand_Q * 2) - 1)
    elif M == 16:
        s = ((-2 * np.fix(rand_I * 4) + 3) + 1j * (-2 * np.fix(rand_Q * 4) + 3)) / np.sqrt(10.0)
    return s

def ofdm_mod(b, N, Fs, Fd):
    sp = b.reshape(N, -1)
    Nfft = int(Fs / Fd * N)
    x1 = np.concatenate((sp[:N // 2], np.zeros((Nfft - N, sp.shape[1]), dtype=complex), sp[N // 2:]))
    t = N * Fs / Fd * np.fft.ifft(x1, axis=0)
    return t.reshape(-1, 1, order='F')                       # 물리적 전송 순서

def ofdm_demod(rx, N, Fs, Fd):
    Nfft = int(Fs / Fd * N)
    sp = ((Fd / Fs) / N * rx).reshape(Nfft, -1, order='F')  # ofdm_mod와 같은 순서로 되돌림
    f = np.fft.fft(sp, axis=0)
    return np.concatenate((f[:N // 2], f[Nfft - N // 2:])).reshape(1, -1)

def decision(D, M, signal_in, signal_out):
    """기존 decision()과 같은 판정·비트오류 계산을 벡터화 (Gray: 부호 비트 + 크기 비트)"""
    scale = np.sqrt(10.0) if M == 16 else 1.0
    r, i = np.real(signal_out) * scale, np.imag(signal_out) * scale
    if M == 16:
        dr = np.where(r > 0, np.where(r > 2, 3, 1), np.where(r < -2, -3, -1))
        di = np.where(i > 0, np.where(i > 2, 3, 1), np.where(i < -2, -3, -1))
    else:
        dr = np.where(r > 0, 1, -1)
        di = np.where(i > 0, 1, -1) if M == 4 else np.zeros_like(dr)
    tr = np.rint(np.real(signal_in) * scale)                 # 송신 레벨 (부동소수 오차 제거)
    ti = np.rint(np.imag(signal_in) * scale)
    sym_err = np.sum((dr != tr) | (di != ti))
    bit_err = np.sum(dr * tr < 0) + np.sum(np.abs(dr) != np.abs(tr))
    if M != 2:
        bit_err += np.sum(di * ti < 0) + np.sum(np.abs(di) != np.abs(ti))
    return sym_err / D, bit_err / (D * np.log2(M))

def evm_pct(rx_sym, tx_sym):
    """이득 정규화 EVM: 최소제곱 복소 이득 g로 나눈 뒤 오차를 잰다 (수신기의 이득·위상 보정)"""
    g = np.vdot(tx_sym, rx_sym) / np.vdot(tx_sym, tx_sym)
    return np.sqrt(np.mean(np.abs(rx_sym / g - tx_sym) ** 2) / np.mean(np.abs(tx_sym) ** 2)) * 100


# ============================================================
# 모델 정의 (학습 파일과 동일한 구조)
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

@tf.keras.utils.register_keras_serializable()
class PositionalEncoding(layers.Layer):
    def __init__(self, sequence_length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length, self.d_model = sequence_length, d_model
        pos = tf.range(sequence_length, dtype=tf.float32)[:, tf.newaxis]
        idx = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
        ang = pos * (1 / tf.pow(10000.0, (2 * (idx // 2)) / tf.cast(d_model, tf.float32)))
        self.pos_encoding = tf.concat([tf.math.sin(ang[:, 0::2]), tf.math.cos(ang[:, 1::2])], -1)[tf.newaxis, ...]
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

def build_transformer_dpd(seq_len, d_model=64, num_heads=4, ff_dim=128, num_layers=2,
                          k_orders=(2, 4), q_depth=3):
    inputs = layers.Input(shape=(seq_len, 2))
    x = layers.Dense(d_model)(PhysicsBasisLayer(k_orders=k_orders, q_depth=q_depth)(inputs))
    x = PositionalEncoding(seq_len, d_model)(x)
    for _ in range(num_layers):
        x = transformer_encoder(x, d_model, num_heads, ff_dim)
    correction = layers.Dense(2, activation="linear")(x)
    return models.Model(inputs, layers.Add()([inputs, correction]), name="Transformer_DPD_DLA")

class TransformerDPD_DLA(models.Model):
    """학습 파일의 래퍼와 같은 구조 (가중치 경로를 맞추기 위해 필요)"""
    def __init__(self, core, **kwargs):
        super().__init__(**kwargs)
        self.core = core
    def call(self, inputs, training=False):
        return self.core(inputs, training=training)

def list_checkpoints(model_dir, prefix):
    """{전체 epoch: 경로}. 커리큘럼 단계의 epoch는 누적 epoch로 바꾼다."""
    pat = re.compile(re.escape(prefix) + r'_stage(\d+)_epoch(\d+)\.weights\.h5')
    off = _mod.stage_epoch_offsets()
    out = {}
    for f in os.listdir(model_dir):
        m = pat.fullmatch(f)
        if m:
            out[off.get(int(m.group(1)), 0) + int(m.group(2))] = os.path.join(model_dir, f)
    if not out:
        raise FileNotFoundError(f"{prefix}_stage*_epoch*.weights.h5 가 {model_dir} 에 없습니다.")
    return dict(sorted(out.items()))

def build_model(model_type):
    """(가중치 로드용 net, 추론용 함수)"""
    x_dummy = tf.zeros((1, seq_len, 2))
    if model_type == 'vtae':
        net = DVAE_Curriculum(build_encoder(seq_len), build_decoder(seq_len), backoff=float(back_off))
        net.build((None, seq_len, 2)); _ = net(x_dummy)
        return net, net
    if model_type == 'transformer_conv':                     # 일반 Transformer: core 가중치만 저장됨
        core = _mod.build_transformer_conventional(seq_len, _mod.EXP_CONV_INPUT)
        _ = core(x_dummy)
        return core, core
    core = build_transformer_dpd(seq_len)
    net  = TransformerDPD_DLA(core)
    net.build((None, seq_len, 2)); _ = core(x_dummy)
    return net, core

def run_dpd(infer, x_time, rms):
    """시간 신호 → DPD 출력 (학습과 같은 RMS 정규화, 연속된 64샘플 시퀀스)"""
    x_seq = (x_time.reshape(-1) / rms).reshape(-1, seq_len)
    x_in  = np.stack([x_seq.real, x_seq.imag], -1).astype(np.float32)
    outs  = [infer(tf.constant(x_in[i:i + 4096]), training=False).numpy() for i in range(0, len(x_in), 4096)]
    y_seq = np.concatenate(outs, axis=0)
    return ((y_seq[..., 0] + 1j * y_seq[..., 1]).reshape(-1) * rms).reshape(-1, 1)

def inband_gain(sig, tx_sym):
    rx = ofdm_demod(sig, N, Fs, Fd)
    return np.vdot(tx_sym, rx) / np.vdot(tx_sym, tx_sym)

def noise_sigma(sig, snr_db):
    p = np.mean(np.abs(sig) ** 2)
    return np.sqrt(p * (Fs / Fd) / 10 ** (snr_db / 10) / 2 / np.log2(M))

def required_snr(ber_curve, snr_axis, target):
    """log-BER 선형 보간으로 목표 BER에 필요한 Eb/N0 (범위 밖이면 nan)"""
    b = np.asarray(ber_curve, dtype=float)
    if target < b.min() or target > b.max():
        return np.nan
    return float(np.interp(-np.log10(target), -np.log10(np.maximum(b, 1e-12)), snr_axis))

def ber_curves(signals, tx_sym, snr_axis, gains, seed=12345):
    """signals: {이름: 시간 신호}. 같은 잡음·페이딩으로 Rayleigh/Rician BER 계산"""
    rng = np.random.default_rng(seed)
    n_sym = tx_sym.size
    res = {ch: {k: np.zeros(len(snr_axis)) for k in signals} for ch in ('rayleigh', 'rician')}
    shape = next(iter(signals.values())).shape
    for ch, K in (('rayleigh', 0), ('rician', 4)):
        for m, snr in enumerate(snr_axis):
            w = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
            h = np.sqrt(K / (1 + K)) + np.sqrt(1 / (1 + K)) * \
                (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)
            for k, s in signals.items():
                rx = (h * s + noise_sigma(s, snr) * w) / h               # 완전 CSI 등화
                r_sym = ofdm_demod(rx, N, Fs, Fd).reshape(-1)
                if GAIN_CORRECT_BER:
                    r_sym = r_sym / gains[k]                              # PA 이득 보정
                _, res[ch][k][m] = decision(n_sym, M, tx_sym.reshape(-1), r_sym)
            del w, h
    return res

def ber_awgn(signals, tx_sym, snr_axis, gains, seed=24680):
    """AWGN BER (페이딩 없음). 페이딩 BER과 다른 난수열을 써서 기존 Rayleigh/Rician 결과는 그대로 유지된다."""
    rng = np.random.default_rng(seed)
    n_sym = tx_sym.size
    res = {k: np.zeros(len(snr_axis)) for k in signals}
    shape = next(iter(signals.values())).shape
    for m, snr in enumerate(snr_axis):
        w = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)     # 방식 공통 잡음
        for k, s in signals.items():
            r_sym = ofdm_demod(s + noise_sigma(s, snr) * w, N, Fs, Fd).reshape(-1)
            if GAIN_CORRECT_BER:
                r_sym = r_sym / gains[k]                                      # PA 이득 보정
            _, res[k][m] = decision(n_sym, M, tx_sym.reshape(-1), r_sym)
        del w
    return res


# ============================================================
# 1. 평가 신호 생성
# ============================================================
b        = gen_mapping(M, D, SEED)
tx_sym   = b.reshape(1, -1)
IQ_out_r = ofdm_mod(b, N, Fs, Fd)
P_ref    = np.mean(np.abs(IQ_out_r) ** 2)                   # PA 기준 전력 (원신호 평균 전력)
print(f"평가 신호: {IQ_out_r.size} 샘플, 복조 확인 {evm_pct(ofdm_demod(IQ_out_r, N, Fs, Fd), tx_sym):.1e} % (0이어야 함)")


# ============================================================
# 2. 모델 로드와 DPD 추론 (마지막 체크포인트)
# ============================================================
tf.keras.backend.clear_session()
rms = np.sqrt(P_ref)
nets, ckpts, signals = {}, {}, {
    'ideal': IQ_out_r,
    'hpa':   polynomial_np(IQ_out_r, back_off, ref_power=P_ref),
}
dpd_out = {}
for mt in MODELS:
    nets[mt] = build_model(mt)
    ckpts[mt] = list_checkpoints(MODEL_DIR, _mod.ckpt_prefix(mt) + LOSS_TAG)
    last_ep, w_path = list(ckpts[mt].items())[-1]
    nets[mt][0].load_weights(w_path)
    print(f"[{mt}] 가중치 로드: {os.path.basename(w_path)} (전체 epoch {last_ep})")
    dpd_out[mt] = run_dpd(nets[mt][1], IQ_out_r, rms)
    signals[mt] = polynomial_np(dpd_out[mt], back_off, ref_power=P_ref)   # PA 기준 전력 고정

# 노이즈 없는 대역 내 EVM과 이득
print("\n노이즈 없는 대역 내 EVM (복조 후 데이터 부반송파)")
evm_nf, gains = {}, {}
for k, s_ in signals.items():
    rx = ofdm_demod(s_, N, Fs, Fd)
    evm_nf[k] = evm_pct(rx, tx_sym)
    gains[k] = np.vdot(tx_sym, rx) / np.vdot(tx_sym, tx_sym)
    print(f"  {k:11s}: EVM {evm_nf[k]:.4f} %, 이득 |g| = {abs(gains[k]):.4f}, 위상 {np.degrees(np.angle(gains[k])):+.2f} deg")


# ============================================================
# 3. AWGN EVM vs Eb/N0 (노이즈 전력은 방식마다 자기 신호 전력 기준)
# ============================================================
rng = np.random.default_rng(12345)
evm_awgn = {k: np.zeros(len(SNR)) for k in signals}
for m, snr in enumerate(SNR):
    w = rng.standard_normal(IQ_out_r.shape) + 1j * rng.standard_normal(IQ_out_r.shape)   # 공통 노이즈
    for k, s_ in signals.items():
        evm_awgn[k][m] = evm_pct(ofdm_demod(s_ + noise_sigma(s_, snr) * w, N, Fs, Fd), tx_sym)
    del w


# ============================================================
# 4. BER (Rayleigh K = 0, Rician K = 4, 완전 CSI)
# ============================================================
ber = ber_curves(signals, tx_sym, SNR, gains, seed=54321)
ber['awgn'] = ber_awgn(signals, tx_sym, SNR, gains, seed=24680)   # AWGN BER 추가


# ============================================================
# 5. 결과 출력과 저장
# ============================================================
fmt = lambda a, p=6: '[' + ', '.join(f'{v:.{p}g}' for v in a) + ']'
print("\n" + "=" * 70 + f"\n결과 ({EXP_TAG}, IBO = {back_off} dB, 이득 보정 BER = {GAIN_CORRECT_BER})\n" + "=" * 70)
print(f"Eb/N0 = {[int(v) for v in SNR]}")
print("노이즈 없는 대역 내 EVM: " + ", ".join(f"{k} {v:.4f} %" for k, v in evm_nf.items() if k != 'ideal'))
for k in signals:
    print(f"evm_{k} = {fmt(evm_awgn[k], 5)};")
for ch in ber:
    for k in signals:
        print(f"ber_{ch}_{k} = {fmt(ber[ch][k])};")

print("\nRician (K = 4) SNR penalty [dB] (이상적 선형 시스템 대비, nan = 범위 밖)")
print("  목표 BER   " + "  ".join(f"{k:>11s}" for k in signals if k != 'ideal'))
for t in TARGET_BER:
    ref = required_snr(ber['rician']['ideal'], SNR, t)
    row = [required_snr(ber['rician'][k], SNR, t) - ref for k in signals if k != 'ideal']
    print(f"  {t:8.0e}  " + "  ".join(f"{v:11.2f}" for v in row))

print("\nAWGN SNR penalty [dB] (이상적 선형 시스템 대비, nan = 범위 밖)")
print("  목표 BER   " + "  ".join(f"{k:>11s}" for k in signals if k != 'ideal'))
for t in TARGET_BER_AWGN:
    ref = required_snr(ber['awgn']['ideal'], SNR, t)
    row = [required_snr(ber['awgn'][k], SNR, t) - ref for k in signals if k != 'ideal']
    print(f"  {t:8.0e}  " + "  ".join(f"{v:11.2f}" for v in row))

out = {'EbN0': SNR.astype(float), 'gain_correct_ber': float(GAIN_CORRECT_BER)}
for k in signals:
    out[f'evm_nf_{k}'] = evm_nf[k]
    out[f'evm_awgn_{k}'] = evm_awgn[k]
    for ch in ber:
        out[f'ber_{ch}_{k}'] = ber[ch][k]
res_path = os.path.join(MODEL_DIR, f'results_{EXP_TAG}{LOSS_TAG}.mat')
savemat(res_path, out)
print(f"\n결과 저장: {res_path}")

if SAVE_AMPM:                                                 # am_pm_plot.m 용 (앞부분 100만 샘플)
    n = min(1_000_000, IQ_out_r.size)
    for mt in MODELS:
        ampm_path = os.path.join(MODEL_DIR, {'vtae': 'dpd_vtae', 'transformer': 'dpd_trans', 'transformer_conv': 'dpd_trans_conv'}[mt] + f'_{EXP_TAG}{LOSS_TAG}.mat')
        savemat(ampm_path, {'x_ref': IQ_out_r[:n].reshape(-1), 'z': dpd_out[mt][:n].reshape(-1)})
        print(f"AM/AM 그림용 저장: {ampm_path}")


# ============================================================
# 6. 학습 곡선: 체크포인트별 EVM과 BER (작은 평가 신호, 두 모델 공통 잡음)
# ============================================================
if EVAL_CURVE:
    Dc   = N * CURVE_BLOCK
    bc   = gen_mapping(M, Dc, SEED + 100)
    txc  = bc.reshape(1, -1)
    xc   = ofdm_mod(bc, N, Fs, Fd)
    Pc   = np.mean(np.abs(xc) ** 2)
    base = {'ideal': xc, 'hpa': polynomial_np(xc, back_off, ref_power=Pc)}
    base_g = {k: inband_gain(v, txc) for k, v in base.items()}
    base_ber = ber_curves(base, txc, CURVE_SNR, base_g, seed=777)
    epochs = sorted(set.intersection(*[set(ckpts[mt]) for mt in MODELS]))
    rows = []
    print("\n학습 곡선 (Rician 10^-2 penalty, 18 dB BER)")
    hdr = "  epoch " + " ".join(f"{mt+' EVM':>15s} {mt+' pen':>15s} {mt+' BER18':>15s}" for mt in MODELS)
    print(hdr)
    ref = required_snr(base_ber['rician']['ideal'], CURVE_SNR, 1e-2)
    for ep in epochs:
        sig, g = {}, {}
        for mt in MODELS:
            nets[mt][0].load_weights(ckpts[mt][ep])
            sig[mt] = polynomial_np(run_dpd(nets[mt][1], xc, np.sqrt(Pc)), back_off, ref_power=Pc)
            g[mt] = inband_gain(sig[mt], txc)
        bc_ = ber_curves(sig, txc, CURVE_SNR, g, seed=777)
        row = {'epoch': ep}
        for mt in MODELS:
            row[f'{mt}_evm'] = evm_pct(ofdm_demod(sig[mt], N, Fs, Fd), txc)
            row[f'{mt}_gain'] = abs(g[mt])
            row[f'{mt}_pen_rician_1e-2'] = required_snr(bc_['rician'][mt], CURVE_SNR, 1e-2) - ref
            for ch in ('rician', 'rayleigh'):
                for snr, v in zip(CURVE_SNR, bc_[ch][mt]):
                    row[f'{mt}_ber_{ch}_{int(snr)}dB'] = v
        rows.append(row)
        print(f"  {ep:5d} " + " ".join(f"{row[mt+'_evm']:15.3f} {row[mt+'_pen_rician_1e-2']:15.2f} "
                                       f"{row[f'{mt}_ber_rician_{int(CURVE_SNR[-1])}dB']:15.3e}" for mt in MODELS))
    import pandas as pd
    curve_path = os.path.join(MODEL_DIR, f'curve_{EXP_TAG}{LOSS_TAG}.csv')
    df = pd.DataFrame(rows)
    for k in base:
        df[f'{k}_pen_rician_1e-2'] = required_snr(base_ber['rician'][k], CURVE_SNR, 1e-2) - ref
    df.to_csv(curve_path, index=False)
    print(f"학습 곡선 저장: {curve_path}  (HPA penalty {required_snr(base_ber['rician']['hpa'], CURVE_SNR, 1e-2) - ref:.2f} dB)")


# ============================================================
# 7. 반복 학습 통계: seed별 노이즈 없는 EVM·이득·Rician penalty → 평균, 표준편차, 95% 신뢰구간
#    (체크포인트가 없는 seed는 건너뜀. 평가 신호·잡음·페이딩은 위와 같음)
# ============================================================
if EVAL_REPEAT:
    from scipy import stats
    import pandas as pd
    seeds = list(_mod.EXP_SEEDS_EVAL)
    seed_saved = _mod.EXP_SEED
    ideal_req = {t: required_snr(ber['rician']['ideal'], SNR, t) for t in TARGET_BER}
    ideal_req_awgn = {t: required_snr(ber['awgn']['ideal'], SNR, t) for t in TARGET_BER_AWGN}
    rep_rows = []
    print("\n반복 학습 평가 (seed: " + ", ".join(str(x) for x in seeds) + ")")
    for mt in MODELS:
        for sd in seeds:
            _mod.EXP_SEED = sd                                   # 체크포인트 이름만 바꾸는 용도
            prefix = _mod.ckpt_prefix(mt) + LOSS_TAG
            try:
                ck = list_checkpoints(MODEL_DIR, prefix)
            except FileNotFoundError:
                print(f"  [{mt}] seed {sd}: 체크포인트 없음 ({prefix}) — 건너뜀")
                continue
            ep, path = list(ck.items())[-1]
            nets[mt][0].load_weights(path)
            sig = polynomial_np(run_dpd(nets[mt][1], IQ_out_r, rms), back_off, ref_power=P_ref)
            rx = ofdm_demod(sig, N, Fs, Fd)
            g = np.vdot(tx_sym, rx) / np.vdot(tx_sym, tx_sym)
            row = {'model': mt, 'seed': 'none' if sd is None else sd, 'epoch': ep,
                   'evm': evm_pct(rx, tx_sym), 'gain': abs(g)}
            if REPEAT_BER:                                       # 위 4절과 같은 잡음·페이딩 (seed=54321)
                b_ = ber_curves({mt: sig}, tx_sym, SNR, {mt: g}, seed=54321)
                for t in TARGET_BER:
                    row[f'pen_{t:.0e}'] = required_snr(b_['rician'][mt], SNR, t) - ideal_req[t]
                row['ber_rician_18dB'] = b_['rician'][mt][-1]
                ba_ = ber_awgn({mt: sig}, tx_sym, SNR, {mt: g}, seed=24680)   # 위 4절과 같은 AWGN 잡음
                for t in TARGET_BER_AWGN:
                    row[f'awgn_pen_{t:.0e}'] = required_snr(ba_[mt], SNR, t) - ideal_req_awgn[t]
            rep_rows.append(row)
            print(f"  [{mt}] seed {row['seed']}: EVM {row['evm']:.4f} %, |g| {row['gain']:.4f}" +
                  ("".join(f", pen({t:.0e}) {row[f'pen_{t:.0e}']:.2f} dB" for t in TARGET_BER) +
                   "".join(f", AWGN pen({t:.0e}) {row[f'awgn_pen_{t:.0e}']:.2f} dB" for t in TARGET_BER_AWGN)
                   if REPEAT_BER else ""))
            del sig, rx
    _mod.EXP_SEED = seed_saved

    df_rep = pd.DataFrame(rep_rows)
    if len(df_rep):
        cols = ['evm', 'gain'] + ([f'pen_{t:.0e}' for t in TARGET_BER] + ['ber_rician_18dB'] +
                                  [f'awgn_pen_{t:.0e}' for t in TARGET_BER_AWGN] if REPEAT_BER else [])
        print("\n반복 학습 요약 (평균 ± 표준편차, [95% 신뢰구간], n = 학습 횟수)")
        for mt, grp in df_rep.groupby('model', sort=False):
            n = len(grp)
            print(f"  {mt} (n = {n})")
            for c in cols:
                v = grp[c].to_numpy(dtype=float)
                m_, sd_ = np.nanmean(v), (np.nanstd(v, ddof=1) if n > 1 else np.nan)
                hw = stats.t.ppf(0.975, n - 1) * sd_ / np.sqrt(n) if n > 1 else np.nan
                print(f"    {c:16s} {m_:.4g} ± {sd_:.2g}   [{m_ - hw:.4g}, {m_ + hw:.4g}]")
        # 두 모델 비교 (Welch t 검정: 등분산 가정 없음)
        mts = list(df_rep['model'].unique())
        if 'vtae' in mts:
            for other in [m for m in mts if m != 'vtae']:
                a_ = df_rep.loc[df_rep.model == 'vtae', 'evm'].to_numpy(float)
                b_ = df_rep.loc[df_rep.model == other, 'evm'].to_numpy(float)
                if len(a_) > 1 and len(b_) > 1:
                    t_, p_ = stats.ttest_ind(a_, b_, equal_var=False)
                    print(f"  EVM 차이 vtae - {other}: {a_.mean() - b_.mean():+.4f} %p "
                          f"({100 * (a_.mean() / b_.mean() - 1):+.1f} %), Welch t = {t_:.2f}, p = {p_:.2g}")
        rep_path = os.path.join(MODEL_DIR, f'repeat_{_mod.base_tag()}{LOSS_TAG}.csv')
        df_rep.to_csv(rep_path, index=False)
        print(f"반복 학습 결과 저장: {rep_path}")
