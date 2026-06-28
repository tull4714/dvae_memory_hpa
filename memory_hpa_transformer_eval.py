import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import csv

#import dec_pse_pbe

from scipy.special import erfc
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers, models # Added for models.Model

import sys
import os

# dvae_curriculum.py에서 polynomial_tf만 로드 (HPA 시뮬레이션용)
# Transformer 모델 평가에는 DVAE 클래스가 불필요
import importlib.util, shutil

module_path = '/content/drive/MyDrive'
if module_path not in sys.path:
    sys.path.append(module_path)

_dvae_path = '/content/drive/MyDrive/NonlinearMemory/dvae_curriculum.py'

# __pycache__ 삭제 (오염된 캐시 제거)
_pycache = '/content/drive/MyDrive/NonlinearMemory/__pycache__'
if os.path.exists(_pycache):
    shutil.rmtree(_pycache)

# 이전 캐시 모듈 제거
for _key in list(sys.modules.keys()):
    if 'dvae_curriculum' in _key:
        del sys.modules[_key]

# 파일 직접 로드
_spec = importlib.util.spec_from_file_location("dvae_curriculum", _dvae_path)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

polynomial_tf     = _mod.polynomial_tf       # HPA 시뮬레이션에 사용
PhysicsBasisLayer = _mod.PhysicsBasisLayer   # 물리 기저 (훈련 모델과 동일)
print("polynomial_tf 로드 성공")


# ==========================================
# Transformer DPD 모델 정의 (Transformer_DPD.py와 동일 구조)
# ==========================================
class PositionalEncoding(layers.Layer):
    """트랜스포머에 시간(순서) 정보를 주입하는 레이어"""
    def __init__(self, sequence_length, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.pos_encoding = self.positional_encoding(sequence_length, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

def transformer_encoder(inputs, d_model, num_heads, ff_dim, dropout=0.1):
    """단일 트랜스포머 블록"""
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inputs, inputs)
    attn_output = layers.Dropout(dropout)(attn_output)
    out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attn_output)
    ffn_output = layers.Dense(ff_dim, activation="relu")(out1)
    ffn_output = layers.Dense(d_model)(ffn_output)
    ffn_output = layers.Dropout(dropout)(ffn_output)
    return layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

def build_transformer_dpd(seq_len, d_model, num_heads, ff_dim, num_layers,
                          k_orders=(2, 4), q_depth=3):
    """
    물리기저 + Transformer + residual 구조 (Transformer_DPD_DLA.py와 동일).

      원본 IQ → PhysicsBasisLayer → Dense(d_model) → PositionalEncoding
              → Transformer × num_layers → Dense(2) = correction
      predistorted = 원본 + correction   (residual)
    """
    inputs = layers.Input(shape=(seq_len, 2))
    basis = PhysicsBasisLayer(k_orders=k_orders, q_depth=q_depth)(inputs)
    x = layers.Dense(d_model)(basis)
    x = PositionalEncoding(seq_len, d_model)(x)
    for _ in range(num_layers):
        x = transformer_encoder(x, d_model, num_heads, ff_dim)
    correction = layers.Dense(2, activation="linear")(x)
    predistorted = layers.Add()([inputs, correction])   # residual
    return models.Model(inputs=inputs, outputs=predistorted, name="Transformer_DPD_DLA")


class TransformerDPD_DLA(models.Model):
    """
    훈련 파일(Transformer_DPD_DLA.py)과 동일한 래퍼.
    가중치가 래퍼 구조(self.core.*)로 저장되었으므로,
    로드 시에도 동일 래퍼로 감싸야 계층 경로가 일치한다.
    """
    def __init__(self, core, backoff=1.0, memory_depth=0, snr_db=-1.0, **kwargs):
        super().__init__(**kwargs)
        self.core          = core
        self.backoff       = backoff
        self._memory_depth = memory_depth
        self._snr_db       = snr_db

    def call(self, inputs, training=False):
        return self.core(inputs, training=training)

def polynomial(p_in, backoff, snr_db=-1.0):
    # Max row index is 5, max col index is 3. So, size 5x3 (or 6x3 for 0-indexing)
    c = np.zeros((6, 3), dtype=complex) # Using 6 rows to accommodate index 5
    c[0, 0] = 1 # c(1,1) in MATLAB is c[0,0] in Python
    c[2, 0] = -0.00542 - 0.02900j # c(3,1) in MATLAB is c[2,0] in Python
    c[4, 0] = -0.009657 - 0.007028j # c(5,1) in MATLAB is c[4,0] in Python
    c[0, 1] = -0.00680 - 0.00023j # c(1,2) in MATLAB is c[0,1] in Python
    c[2, 1] = 0.02234 + 0.02317j # c(3,2) in MATLAB is c[2,1] in Python
    c[4, 1] = -0.002451 - 0.003735j # c(5,2) in MATLAB is c[4,1] in Python
    c[0, 2] = 0.00289 - 0.00054j # c(1,3) in MATLAB is c[0,2] in Python
    c[2, 2] = -0.00621 - 0.00932j # c(3,3) in MATLAB is c[2,2] in Python
    c[4, 2] = 0.001229 + 0.001508j # c(5,3) in MATLAB is c[4,2] in Python

    # Ensure p_in is a 1D numpy array of complex numbers for consistent element-wise operations
    p_in_flat = np.asarray(p_in, dtype=complex).flatten()
    len_p_in = len(p_in_flat)
    p_out_flat = np.zeros(len_p_in, dtype=complex) # Initialize as 1D array

    #======================== Normalized =========================%
    data_tx_avg = np.sum(np.abs(p_in_flat)**2) / len_p_in # power, scalar
    tx_amp_flat = np.sqrt( (np.abs(p_in_flat)**2) / (data_tx_avg * (10**(backoff/10))) ) # 1D array
    #=============================================================%
    for n_idx in range(len_p_in):
        for k_idx in [0, 2, 4]:
            for q_idx in range(3):
                if (n_idx - q_idx) >= 0:
                    x = tx_amp_flat[n_idx - q_idx] # x is scalar
                else:
                    x = 0

                p_out_flat[n_idx] += c[k_idx, q_idx] * x * (np.abs(x)) ** (k_idx) # p_out_flat[n_idx] is scalar assignment
    # At this point, p_out_flat is (len_p_in,)

    # Final calculations
    # All operations now involve 1D arrays or scalars, preventing implicit outer product
    hpa_data_flat = p_out_flat * np.exp(np.angle(p_in_flat) * 1j) * np.sqrt(data_tx_avg * (10**(backoff/10)))

    hpa_data_tx_avg = np.sum(np.abs(hpa_data_flat)**2) / len(hpa_data_flat) # power, scalar
    hpa_data_tx_amp = np.sqrt( (np.abs(hpa_data_flat)**2) / (hpa_data_tx_avg * (10**(backoff/10)))) # 1D array

    # Add AWGN
    if snr_db >= 0.0:
        snr_linear = 10**(snr_db / 10)

        # Convert numpy floats to TensorFlow float64 tensors to ensure consistent dtypes
        hpa_data_tx_avg_tf = tf.constant(hpa_data_tx_avg, dtype=tf.float64)
        snr_linear_tf = tf.constant(snr_linear, dtype=tf.float64)

        noise_power = hpa_data_tx_avg_tf / snr_linear_tf
        noise_std = tf.sqrt(noise_power / 2) # This will now be tf.float64

        # Generate noise with dtype=tf.float64 to match noise_std
        noise_shape = tf.shape(hpa_data_flat) # Get shape from hpa_data_flat
        n_real = tf.random.normal(shape=noise_shape, mean=0.0, stddev=noise_std, dtype=tf.float64)
        n_imag = tf.random.normal(shape=noise_shape, mean=0.0, stddev=noise_std, dtype=tf.float64)

        # Form complex noise (tf.complex128)
        n = tf.complex(n_real, n_imag)

        # Convert hpa_data_flat (numpy complex128) to tf.complex128 for addition
        hpa_data_flat_tf = tf.constant(hpa_data_flat, dtype=tf.complex128)

        # Perform addition and convert back to numpy complex128
        hpa_data_flat = (hpa_data_flat_tf + n).numpy()

    # Return as a column vector (N, 1) if the original p_in was a column vector, otherwise return 1D
    # To be consistent with the original context where IQ_out_r is (N,1) and dvae_r is (N,),
    # it's safer to always return as (N,1) as the code seems to expect.
    # However, if the caller needs a (N,) array for the second returned value (hpa_data_tx_amp),
    # that needs to be clarified or adjusted in the calling code.
    # For now, let's make sure hpa_data is (N,1) as it seems to be the primary output.

    # The original line 481 in the traceback suggests hpa_dvae_r is used directly for polynomial,
    # which was (N,). So the function can return (N,) array too, depending on usage.
    # Given the memory error on (N,N), the safest is to ensure everything is 1D until the very end.

    return hpa_data_flat.reshape(-1, 1), hpa_data_tx_avg, hpa_data_tx_amp.reshape(-1, 1)

# 훈련 시와 동일한 정규화 함수 추가
def normalize_with_rms(I_data, Q_data):
    """RMS 기반 정규화로 더 안정적인 정규화"""
    magnitude = np.sqrt(I_data**2 + Q_data**2)
    rms_magnitude = np.sqrt(np.mean(magnitude**2))
    if rms_magnitude > 0:
        I_data_normalized = I_data / rms_magnitude
        Q_data_normalized = Q_data / rms_magnitude
    else:
        I_data_normalized = I_data
        Q_data_normalized = Q_data
    return I_data_normalized, Q_data_normalized, rms_magnitude

def denormalize_with_rms(I_data, Q_data, rms_magnitude):
    """RMS 기반 역정규화"""
    return I_data * rms_magnitude, Q_data * rms_magnitude

def decision(D, M, signal_in, signal_out):
    """
    수신된 신호를 복조하고 심볼 오류율(Pse) 및 비트 오류율(Pbe)을 계산합니다.

    16QAM은 gen_mapping에서 sqrt(10)으로 정규화되어 심볼 레벨이
    ±3/√10, ±1/√10 이므로, 경판정 전에 sqrt(10)을 곱해 원래 레벨
    (±3, ±1)로 복원한 뒤 임계값 ±2로 판정한다.
    """

    numoferror = 0
    numoferror_b = 0

    if M > 1:
        log2M = np.log2(M)
    else:
        log2M = 1

    # 16QAM 정규화 해제 스케일
    qam_scale = np.sqrt(10.0) if M == 16 else 1.0

    for k in range(D):

        r_out = np.real(signal_out[k]) * qam_scale
        i_out = np.imag(signal_out[k]) * qam_scale

        # --- 1. 실수부 (I) 결정 ---
        decis_real = 0
        if M == 2 or M == 4:
            if r_out > 0:
                decis_real = 1
            else:
                decis_real = -1
        elif M == 16:
            if r_out > 0:
                if r_out > 2:
                    decis_real = 3
                else:
                    decis_real = 1
            else:
                if r_out < -2:
                    decis_real = -3
                else:
                    decis_real = -1

        # --- 2. 허수부 (Q) 결정 ---
        decis_imag = 0
        if M == 2:
            decis_imag = 0
        elif M == 4:
            if i_out > 0:
                decis_imag = 1j
            else:
                decis_imag = -1j
        elif M == 16:
            if i_out > 0:
                if i_out > 2:
                    decis_imag = 3j
                else:
                    decis_imag = 1j
            else:
                if i_out < -2:
                    decis_imag = -3j
                else:
                    decis_imag = -1j

        # --- 3. 최종 결정된 심볼 ---
        decis = decis_real + decis_imag

        # 송신 심볼도 동일 스케일로 복원
        sig_in_k = signal_in[k] * qam_scale

        # --- 4. 심볼 오류 계산 (Pse) ---
        if decis != sig_in_k:
            numoferror += 1

        # --- 5. 비트 오류 계산 (Pbe) ---
        r_in = np.real(sig_in_k)
        i_in = np.imag(sig_in_k)

        # A. 실수부 (I) 오류 계산:

        # 1. 부호 오류 (decis_real * r_in < 0)
        if decis_real * r_in < 0:
            numoferror_b += 1

        # 2. 크기 오류 (abs(decis_real) != abs(r_in))
        if abs(decis_real) != abs(r_in):
            numoferror_b += 1

        # B. 허수부 (Q) 오류 계산:
        if M != 2: # BPSK가 아닐 때만 Q축 오류를 고려

            # 1. 부호 오류 (imag(decis) * i_in < 0)
            if np.imag(decis) * i_in < 0:
                numoferror_b += 1

            # 2. 크기 오류 (abs(np.imag(decis)) != abs(i_in))
            if abs(np.imag(decis)) != abs(i_in):
                numoferror_b += 1

    # 최종 오류율 계산
    Pse = numoferror / D
    if M > 1:
        Pbe = numoferror_b / (D * log2M)
    else:
        Pbe = 0

    return Pse, Pbe

def gen_mapping(M, D):
    np.random.seed(1)
    rand_I = np.random.rand(1, D)
    rand_Q = np.random.rand(1, D)
    s = None

    if M == 2:
        s = 2 * np.fix(rand_I * 2) - 1
    elif M == 4:
        s = (2 * np.fix(rand_I * 2) - 1) + 1j * (2 * np.fix(rand_Q * 2) - 1)
    elif M == 16:
        # 16QAM: I, Q ∈ {-3,-1,+1,+3}, sqrt(10) 단위전력 정규화 (데이터 생성과 동일)
        s = (-2 * np.fix(rand_I * 4) + 3) + 1j * (-2 * np.fix(rand_Q * 4) + 3)
        s = s / np.sqrt(10.0)

    return s

# Convert to real/imag 2-channel for model: shape (N, seq_len, 2)
def to_2ch(z):
    return np.stack([np.real(z), np.imag(z)], axis=-1).astype(np.float32)

# convert to 1ch data from I, Q
def to_1ch(i, q):
    return np.stack([i, q], axis=-1).astype(np.float32)

# convert to complex
def to_complex(x2ch):
    return x2ch[...,0] + 1j * x2ch[...,1]

def cos_sampling(X, Fq, C, upsilon):
    a = np.empty(X)

    for t in range(X):
        a[t] = (1 + upsilon / 2) * np.cos((2 * Fq * np.pi * t) / X + np.pi * C / 360)

    return a.reshape(1, -1)

def sin_sampling(X, Fq, C, upsilon):
    a = np.empty(X)

    for t in range(X):
        a[t] = (1 - upsilon / 2) * np.sin((2 * Fq * np.pi * t) / X - np.pi * C / 360)

    return a.reshape(1, -1)

def cos_predistor(data, X, Fq, epsilon, phi):
    # Amplitude compensation
    a_Ir = 2 / (2 + epsilon) * data

    # Phase compensation
    m = np.arange(0, X)
    S_IC = 1 / (np.cos((2 * Fq * np.pi * m) / X + np.pi * phi / 360))

    I_r = a_Ir * S_IC

    return I_r

def sin_predistor(data, X, Fq, epsilon, phi):
    # Amplitude compensation
    a_Qr = 2 / (2 - epsilon) * data

    # Phase compensation
    m = np.arange(0, X)
    S_QC = 1 / (np.sin((2 * Fq * np.pi * m) / X - np.pi * phi / 360))

    Q_r = a_Qr * S_QC

    return Q_r

def hard_decision(in_vector, M):
    if M == 2:
        out_vector = np.sign(np.real(in_vector))
    elif M == 4:
        r_part = np.real(in_vector)
        i_part = np.imag(in_vector)
        out_vector = np.sign(r_part) + 1j * np.sign(i_part)

    return out_vector

def _qam16_axis_bits(level):
    """
    16QAM 한 축(정규화 전 레벨 -3,-1,+1,+3)의 2비트 그레이 코드 반환.
    -3 → (0,0), -1 → (0,1), +1 → (1,1), +3 → (1,0)
    """
    if level < -2:      # -3
        return (0, 0)
    elif level < 0:     # -1
        return (0, 1)
    elif level < 2:     # +1
        return (1, 1)
    else:               # +3
        return (1, 0)

def _qam16_axis_decide(val):
    """수신값(정규화 전 스케일)을 가장 가까운 16QAM 레벨로 경판정."""
    if val < -2:
        return -3
    elif val < 0:
        return -1
    elif val < 2:
        return 1
    else:
        return 3

def ber_call_qpsk(a, b, M):
    numoferr = 0
    NumOfBitError = 0
    NumOfSymbolError = 0
    SymbolError = 0
    integral = 0

    A = a.flatten()
    B = b.flatten()
    D = np.size(A) if isinstance(A, np.ndarray) else len(A)

    if M == 2:
        ber = np.sum(~(B == A)) / D
    elif M == 4:
        for q in range(D):
            integral_I = np.real(A[q])
            integral_Q = np.imag(A[q])
            B_I = np.real(B[q])
            B_Q = np.imag(B[q])

            decision_I = 1 if integral_I > 0 else -1
            decision_Q = 1 if integral_Q > 0 else -1
            if decision_I != B_I:
                NumOfBitError += 1
                SymbolError = 1
            if decision_Q != B_Q:
                NumOfBitError += 1
                SymbolError = 1
            if SymbolError == 1:
                NumOfSymbolError += 1
            SymbolError = 0

        pb = NumOfBitError / (2 * D)
        ps = NumOfSymbolError / D
        ber = pb

    elif M == 16:
        # 16QAM: 심볼당 4비트 (I축 2비트 + Q축 2비트, 그레이 코딩)
        # A(수신), B(송신) 모두 sqrt(10)으로 정규화되어 있으므로 복원
        scale = np.sqrt(10.0)
        for q in range(D):
            # 정규화 해제 → 원래 레벨 스케일(-3,-1,+1,+3)
            rx_I = np.real(A[q]) * scale
            rx_Q = np.imag(A[q]) * scale
            tx_I = np.real(B[q]) * scale
            tx_Q = np.imag(B[q]) * scale

            # 송신 비트 (정확한 레벨에서)
            tx_I_lvl = _qam16_axis_decide(tx_I)
            tx_Q_lvl = _qam16_axis_decide(tx_Q)
            tx_bits = _qam16_axis_bits(tx_I_lvl) + _qam16_axis_bits(tx_Q_lvl)

            # 수신 비트 (경판정 후)
            rx_I_lvl = _qam16_axis_decide(rx_I)
            rx_Q_lvl = _qam16_axis_decide(rx_Q)
            rx_bits = _qam16_axis_bits(rx_I_lvl) + _qam16_axis_bits(rx_Q_lvl)

            # 4비트 비교
            bit_err = sum(1 for tb, rb in zip(tx_bits, rx_bits) if tb != rb)
            NumOfBitError += bit_err
            if bit_err > 0:
                NumOfSymbolError += 1

        pb = NumOfBitError / (4 * D)   # 심볼당 4비트
        ps = NumOfSymbolError / D
        ber = pb

    return ber

# -----------------------------
# Transformer DPD 하이퍼파라미터 (Transformer_DPD.py와 동일해야 함)
# -----------------------------
d_model    = 64
num_heads  = 4
ff_dim     = 128
num_layers = 2

Fs = 4          # frequency of the sample
Fd = 1          # frequency of the data
N = 64
seq_len = N
beta_kl = 1e-3
back_off = 1    # ★ 데이터 생성/훈련과 동일하게 1로 설정 (16QAM)
Block = 100000
M = 16          # ★ 16QAM (이전: QPSK M=4)
D = N * Block
hpa_snr = 20

# ★ back_off 값 확인 (정의 직후)
print("\n" + "#"*55)
print(f"# back_off 정의값: {back_off}")
print(f"# (훈련 데이터 back_off와 일치해야 함)")
print("#"*55 + "\n")

np.random.seed(seed=int(time.time()))

SNR = np.arange(0, 20, 2)
# Removed org_ber and ber as they are not used
theo_ber_awgn = np.zeros(len(SNR)) # Theoretical BER for AWGN
theo_ber_rayleigh = np.zeros(len(SNR)) # Theoretical BER for Rayleigh (P-CSI)
# Removed errs and errb, replaced by simulated_bers dictionary

b = gen_mapping(M, D)

# 송신기
sp = b.reshape(N, -1)

# Calculate the number of zeros to insert
num_zeros_to_insert = int(Fs / Fd * N - N)

# Handle the case where num_zeros_to_insert might be negative or zero
if num_zeros_to_insert < 0:
    print("Warning: The number of zeros to insert is negative. Adjusting to 0.")
    num_zeros_to_insert = 0

# Create the zero padding
zeros_padding = np.zeros((num_zeros_to_insert, sp.shape[1]), dtype=sp.dtype) # Use dtype of x for consistency


# Concatenate the parts
x1 = np.concatenate((sp[:N//2, :], zeros_padding, sp[N//2:, :]))

ifft_out = np.fft.ifft(x1, axis=0)
ps = ifft_out.reshape(-1, 1)

IQ_out_r = N*Fs/Fd* ps

dvae_i_norm, dvae_q_norm, rms_mag = normalize_with_rms(np.real(IQ_out_r), np.imag(IQ_out_r))
dvae_i_seq = dvae_i_norm.reshape(-1, N)
dvae_q_seq = dvae_q_norm.reshape(-1, N)
dvae_1ch = to_1ch(dvae_i_seq, dvae_q_seq)

# =======================================================================
# 확인 2: dvae_1ch shape 및 reshape 방식 검증
# =======================================================================
print("\n" + "#"*60)
print("# 확인 2: 추론 입력 dvae_1ch 구성 검증")
print("#"*60)
print(f"IQ_out_r shape:        {IQ_out_r.shape}")
print(f"dvae_i_norm shape:     {dvae_i_norm.shape}")
print(f"dvae_i_seq shape:      {dvae_i_seq.shape}  (reshape(-1, {N}))")
print(f"dvae_1ch shape:        {dvae_1ch.shape}  (예상: (num_seq, {N}, 2))")

# =======================================================================
# 확인 1: 추론 입력 vs 훈련 입력 통계 비교
# 훈련에 쓴 input_target CSV를 같은 방식으로 로드/정규화하여 비교
# 두 통계가 다르면 정규화 또는 데이터 생성 방식 불일치
# =======================================================================
print("\n" + "#"*60)
print("# 확인 1: 추론 입력 vs 훈련 입력(input_target) 통계 비교")
print("#"*60)

# --- 추론 입력 통계 ---
print("\n[추론 입력 dvae_1ch]")
print(f"  전체 평균:      {np.mean(dvae_1ch):.6f}")
print(f"  전체 표준편차:  {np.std(dvae_1ch):.6f}")
print(f"  I채널 RMS:      {np.sqrt(np.mean(dvae_1ch[...,0]**2)):.6f}")
print(f"  Q채널 RMS:      {np.sqrt(np.mean(dvae_1ch[...,1]**2)):.6f}")
print(f"  전체 RMS:       {np.sqrt(np.mean(dvae_1ch**2)):.6f}")
print(f"  최댓값:         {np.max(np.abs(dvae_1ch)):.6f}")

# --- 훈련 입력 통계 (input_target CSV를 동일 방식으로 처리) ---
try:
    _tgt_I = pd.read_csv('/content/drive/MyDrive/input_target_I.csv', header=None).to_numpy()
    _tgt_Q = pd.read_csv('/content/drive/MyDrive/input_target_Q.csv', header=None).to_numpy()
    _tgt_I = _tgt_I.reshape(-1, N)
    _tgt_Q = _tgt_Q.reshape(-1, N)

    # 훈련과 동일하게 train 부분의 RMS로 정규화
    _n_train = int(0.8 * _tgt_I.shape[0])
    _train_I = _tgt_I[:_n_train]
    _train_Q = _tgt_Q[:_n_train]
    _, _, _rms_tr = normalize_with_rms(_train_I, _train_Q)

    _tgt_I_norm = _tgt_I / _rms_tr
    _tgt_Q_norm = _tgt_Q / _rms_tr
    _train_1ch = to_1ch(_tgt_I_norm[:_n_train], _tgt_Q_norm[:_n_train])

    print("\n[훈련 입력 input_target (train RMS 정규화)]")
    print(f"  rms_train:      {_rms_tr:.6f}  ← 추론 rms_mag({rms_mag:.6f})와 비교")
    print(f"  전체 평균:      {np.mean(_train_1ch):.6f}")
    print(f"  전체 표준편차:  {np.std(_train_1ch):.6f}")
    print(f"  I채널 RMS:      {np.sqrt(np.mean(_train_1ch[...,0]**2)):.6f}")
    print(f"  Q채널 RMS:      {np.sqrt(np.mean(_train_1ch[...,1]**2)):.6f}")
    print(f"  전체 RMS:       {np.sqrt(np.mean(_train_1ch**2)):.6f}")
    print(f"  최댓값:         {np.max(np.abs(_train_1ch)):.6f}")

    print("\n[핵심 비교]")
    print(f"  rms 비율 (추론/훈련): {rms_mag / _rms_tr:.4f}  (1.0이어야 정상)")
    print(f"  RMS 차이: 추론 {np.sqrt(np.mean(dvae_1ch**2)):.4f} "
          f"vs 훈련 {np.sqrt(np.mean(_train_1ch**2)):.4f}")
    if abs(rms_mag / _rms_tr - 1.0) > 0.05:
        print("  ⚠ rms_mag와 rms_train이 5% 이상 차이남 → 정규화 스케일 불일치!")
    else:
        print("  ✓ 정규화 스케일 일치")
except Exception as e:
    print(f"\n훈련 데이터 비교 실패: {e}")

print("#"*60 + "\n")

keras.backend.clear_session()

# Transformer DLA 가중치 파일 (최신 stage·epoch 자동 탐색)
# 훈련 파일이 transformer_dla_stage{N}_epoch{M}.weights.h5 형식으로 저장
import re as _re
_model_dir = '/content/drive/MyDrive/NonlinearMemory/'
_pat = _re.compile(r'transformer_dla_stage(\d+)_epoch(\d+)\.weights\.h5')
_cands = []
for f in os.listdir(_model_dir):
    m = _pat.match(f)
    if m:
        _cands.append((int(m.group(1)), int(m.group(2)), os.path.join(_model_dir, f)))

if _cands:
    _cands.sort(key=lambda t: (t[0], t[1]))   # (stage, epoch) 순 정렬
    _stage, _epoch, name = _cands[-1]
    print(f"최신 Transformer DLA 가중치 발견: stage {_stage}, epoch {_epoch}")
else:
    name = os.path.join(_model_dir, 'transformer_dla_stage2_epoch40.weights.h5')
    print(f"경고: 저장된 가중치 미발견, 기본 경로 사용: {name}")

# Transformer 모델 생성 후 가중치 로드 (래퍼로 감싸 저장 구조와 일치)
# d_model, num_heads, ff_dim, num_layers, k_orders, q_depth는 훈련과 일치해야 함
_core = build_transformer_dpd(seq_len, d_model, num_heads, ff_dim, num_layers,
                              k_orders=(2, 4), q_depth=3)
_wrapper = TransformerDPD_DLA(_core, backoff=float(back_off),
                              memory_depth=2, snr_db=-1.0)
_wrapper.build((None, seq_len, 2))

# forward pass로 가중치 완전 생성 후 로드
_ = _core(tf.constant(dvae_1ch[:N]), training=False)
_wrapper.load_weights(name)

# 추론에는 내부 core를 직접 사용 (predistorted 출력)
transformer = _core
print(f'Transformer 가중치를 {name}에서 성공적으로 로드했습니다.')

# =======================================================================
# 진단 A: 가중치 로드 검증
# =======================================================================
print("\n" + "%"*60)
print("% 진단 A: 가중치 로드 검증 (레이어별 첫 가중치)")
print("%"*60)
for layer in transformer.layers:
    if len(layer.weights) > 0:
        w = layer.weights[0].numpy()
        print(f"  {layer.name:32s} shape={str(w.shape):18s} "
              f"mean={w.mean():+.4f} std={w.std():.4f}")
print("  (std가 0이거나 모두 동일하면 로드 실패)")
print("%"*60 + "\n")

# =======================================================================
# 진단 B: 배치 분할 효과 확인
# Transformer는 시퀀스 전체에 attention하므로 블록 분할 영향이 클 수 있음
# =======================================================================
print("%"*60)
print("% 진단 B: 배치 분할 효과")
print("%"*60)
_test = dvae_1ch[:100]
_outA = []
for i in range(0, len(_test), N):
    _outA.append(transformer(tf.constant(_test[i:i+N]), training=False).numpy())
_outA = np.concatenate(_outA, axis=0)
_outB = transformer(tf.constant(_test), training=False).numpy()
_diff = np.max(np.abs(_outA - _outB))
print(f"  방식A(64씩) vs 방식B(통째) 최대 차이: {_diff:.8f}")
if _diff < 1e-5:
    print("  ✓ 배치 분할 무관")
else:
    print("  ⚠ 배치 분할이 결과를 바꿈")
print("%"*60 + "\n")

# =======================================================================
# 진단 1: 추론 입력 단계 RMS 확인
# =======================================================================
print("\n" + "="*55)
print("추론 파이프라인 진단")
print("="*55)
print(f"추론 rms_mag (정규화 기준):  {rms_mag:.6f}")
print(f"IQ_out_r RMS:               {np.sqrt(np.mean(np.abs(IQ_out_r)**2)):.6f}")
print(f"입력 RMS:                   {np.sqrt(np.mean(dvae_1ch**2)):.6f}  (정상: ~1.0)")

# Transformer 추론: 원본 신호 → 보정된 신호 (직접 출력)
# Transformer는 correction이 아니라 보정 신호 자체를 출력
sig_in_Linear_list = []
for i in range(0, len(dvae_1ch), N):
    batch = tf.constant(dvae_1ch[i:i+N])
    corrected = transformer(batch, training=False)   # 직접 출력
    sig_in_Linear_list.append(corrected.numpy())
sig_in_Linear = np.concatenate(sig_in_Linear_list, axis=0)

# =======================================================================
# 진단 2: 모델 출력 크기 확인
# =======================================================================
correction_2ch = sig_in_Linear - dvae_1ch   # 입력 대비 변화량
print(f"Transformer 출력 RMS:       {np.sqrt(np.mean(sig_in_Linear**2)):.6f}  (정상: ~1.0)")
print(f"입력 대비 변화량 RMS:       {np.sqrt(np.mean(correction_2ch**2)):.6f}")

sig_in_Linear_c = to_complex(sig_in_Linear)
sp_dvae_norm = sig_in_Linear_c.reshape(-1, 1)

# 역정규화 (원본 신호 RMS 기준)
dvae_i_r, dvae_q_r = denormalize_with_rms(np.real(sp_dvae_norm).flatten(),
                                            np.imag(sp_dvae_norm).flatten(),
                                            rms_mag)
dvae_r = (dvae_i_r + 1j * dvae_q_r).reshape(-1, 1)

# =======================================================================
# 진단 3: 역정규화 후 신호 크기 확인
# =======================================================================
print(f"보정신호 (역정규화 후) RMS: {np.sqrt(np.mean(np.abs(dvae_r)**2)):.6f}")
print(f"  → IQ_out_r RMS와 비슷해야 정상")
print("="*55 + "\n")

# BER 평가 시에는 채널 루프의 노이즈만 사용
# HPA 단계에서는 노이즈를 넣지 않음 (snr_db=-1.0)

# ★ polynomial 호출 직전 back_off 최종 확인
print(f"\n>>> polynomial() 호출 직전 back_off = {back_off} <<<\n")

hpa_data_tx, _, _ = polynomial(IQ_out_r, back_off, -1.0)
hpa_dvae_r, _, _ = polynomial(dvae_r, back_off, -1.0)     # Transformer 보정

# -----------------------------------------------------------------------
# EVM (Error Vector Magnitude) 평가
#
# DPD의 실제 효과는 BER보다 EVM에서 더 명확하게 드러난다.
#   EVM = sqrt( mean(|측정 - 이상|²) / mean(|이상|²) ) × 100 (%)
#
# 판정 기준:
#   evm_dvae < evm_hpa  → DVAE가 HPA 왜곡을 보상 (성공)
#   evm_dvae > evm_hpa  → DVAE가 오히려 신호를 손상 (실패)
# -----------------------------------------------------------------------
def calc_evm(ideal, measured):
    ideal_f = np.asarray(ideal).flatten()
    meas_f  = np.asarray(measured).flatten()
    return np.sqrt(np.mean(np.abs(meas_f - ideal_f) ** 2) /
                   np.mean(np.abs(ideal_f) ** 2)) * 100.0

# 노이즈/페이딩 없는 순수 신호 레벨에서 EVM 측정
# (원본 IQ_out_r 기준 — 정규화 스케일 동일하므로 직접 비교 가능)
evm_hpa  = calc_evm(IQ_out_r, hpa_data_tx)   # 보상 없음: HPA(원본) vs 원본
evm_dvae = calc_evm(IQ_out_r, hpa_dvae_r)    # DVAE 보상: HPA(predistorted) vs 원본

print("\n" + "="*55)
print("EVM 평가 (노이즈/페이딩 제외, HPA 왜곡만)")
print("="*55)
print(f"EVM (보상 없음, HPA):  {evm_hpa:.4f}%")
print(f"EVM (DVAE 보상):       {evm_dvae:.4f}%")
if evm_dvae < evm_hpa:
    improvement = (evm_hpa - evm_dvae) / evm_hpa * 100.0
    print(f"→ DVAE가 EVM을 {improvement:.1f}% 개선 (보상 성공)")
else:
    degradation = (evm_dvae - evm_hpa) / evm_hpa * 100.0
    print(f"→ DVAE가 EVM을 {degradation:.1f}% 악화 (보상 실패)")
print("="*55 + "\n")

sigpwr = np.linalg.norm(IQ_out_r) ** 2 / len(IQ_out_r)

# =======================================================================
# AWGN-only EVM vs SNR (페이딩 없음 — 순수 DPD + AWGN 성능)
# 페이딩(h) 미적용 → 노이즈 증폭(n/h) 없는 순수 AWGN. 채널 무관, 1회 계산.
# =======================================================================
def _ofdm_demod_freq(time_sig):
    rx = (Fd / Fs) / N * time_sig
    sp = rx.reshape(int(Fs / Fd * N), -1)
    fout = np.fft.fft(sp, axis=0)
    s1 = int(Fs / Fd * N - N / 2)
    s2 = int(Fs / Fd * N)
    yy = np.concatenate((fout[: N//2, :], fout[s1:s2, :]))
    return yy.reshape(1, -1).flatten()

awgn_hpa_evm  = np.zeros(len(SNR))
awgn_dvae_evm = np.zeros(len(SNR))
_tx_sym = b.flatten()
_ref_pow = np.mean(np.abs(_tx_sym) ** 2)

np.random.seed(12345)
for m in range(len(SNR)):
    snr_wp = 10 ** (SNR[m] / 10)
    sgma = np.sqrt(sigpwr * (Fs/Fd) / snr_wp / 2 / np.log2(M))
    n = sgma * np.random.randn(*IQ_out_r.shape) + 1j * sgma * np.random.randn(*IQ_out_r.shape)
    rx_hpa_sym  = _ofdm_demod_freq(hpa_data_tx + n)
    rx_dvae_sym = _ofdm_demod_freq(hpa_dvae_r  + n)
    awgn_hpa_evm[m]  = np.sqrt(np.mean(np.abs(rx_hpa_sym  - _tx_sym)**2) / _ref_pow) * 100.0
    awgn_dvae_evm[m] = np.sqrt(np.mean(np.abs(rx_dvae_sym - _tx_sym)**2) / _ref_pow) * 100.0

print("\n" + "="*55)
print("AWGN-only EVM vs SNR (페이딩 없음, 순수 DPD+AWGN)")
print("="*55)
print(f"SNR(dB):           {np.array2string(SNR, precision=0)}")
print(f"HPA  EVM(%):       {np.array2string(awgn_hpa_evm,  precision=3, separator=', ')}")
print(f"Transformer EVM(%): {np.array2string(awgn_dvae_evm, precision=3, separator=', ')}")
print("="*55 + "\n")

# Define channel k-factor values: 0 for Rayleigh, 4 for Rician
channel_k_values = {0: 'Rayleigh', 4: 'Rician'}
simulated_bers = {} # Dictionary to store BERs for each K-factor
hpa_bers = {}
dvae_bers = {}

for k_factor, channel_name in channel_k_values.items():
    current_errb_array = np.zeros(len(SNR)) # Temporary array for current channel's BER
    hpa_errb_array = np.zeros(len(SNR)) # Temporary array for current channel's BER
    dvae_errb_array = np.zeros(len(SNR)) # DVAE BER 배열 초기화

    print(f"Simulating for K-factor = {k_factor} ({channel_name} Fading) with Perfect CSI")

    for m in range(len(SNR)):
        snr_wp = 10 ** (SNR[m] / 10)
        sgma = np.sqrt(sigpwr * (Fs/Fd) / snr_wp / 2 / np.log2(M))

        # Generate complex noise
        n_real = sgma * np.random.randn(*IQ_out_r.shape)
        n_imag = sgma * np.random.randn(*IQ_out_r.shape)
        n = n_real + 1j * n_imag

        # Fading (Using current_k_factor)
        frame = IQ_out_r.size
        if k_factor == 0: # Rayleigh fading (no line-of-sight component)
            h = np.sqrt(1 / (1 + k_factor)) * ((np.random.randn(frame, 1) + 1j * np.random.randn(frame, 1)) / np.sqrt(2))
        else: # Rician fading
            h = np.sqrt(k_factor / (1 + k_factor)) * np.ones((frame, 1)) + np.sqrt(1 / (1 + k_factor)) * ((np.random.randn(frame, 1) + 1j * np.random.randn(frame, 1)) / np.sqrt(2))

        # Apply fading and noise
        receive_data = h * IQ_out_r + n
        hpa_data_rx = h * hpa_data_tx + n
        hpa_data_rx_dvae = h * hpa_dvae_r + n # DVAE

        # !!! Crucial Change: Perfect Channel State Information (P-CSI) Equalization
        # Divide by h to compensate for fading, assuming perfect channel knowledge.
        # This is necessary for demodulation to work well with fading.
        receive_data = receive_data / h
        hpa_data_rx = hpa_data_rx / h
        hpa_data_rx_dvae = hpa_data_rx_dvae / h # DVAE

        ## Receiver
        # Python: Perform FFT and scalar multiplication
        x1 = (Fd / Fs) / N * receive_data
        hpa_rx = (Fd / Fs) / N * hpa_data_rx
        hpa_rx_dvae = (Fd / Fs) / N * hpa_data_rx_dvae  # DVAE

        rx_sp_r = x1.reshape(int(Fs / Fd * N), -1) # Use equalized data
        hpa_sp = hpa_rx.reshape(int(Fs / Fd * N), -1) # Use equalized data
        hpa_sp_dvae = hpa_rx_dvae.reshape(int(Fs / Fd * N), -1) # DVAE

        # Calculate BER for the current channel type
        fft_out_r = np.fft.fft(rx_sp_r, axis=0)
        fft_out_hpa = np.fft.fft(hpa_sp, axis=0)
        fft_out_hpa_dvae = np.fft.fft(hpa_sp_dvae, axis=0) # DVAE

        # Python: Split x1 into parts and concatenate
        # Calculate start and end indices for the second part
        # MATLAB 1-based index (Fs/Fd*N - N/2 + 1) becomes Python 0-based index (int(Fs/Fd*N - N/2))
        start_idx_second_part = int(Fs / Fd * N - N / 2)
        # MATLAB end index (Fs/Fd*N) becomes Python exclusive index (int(Fs / Fd * N))
        end_idx_second_part = int(Fs / Fd * N)

        # Concatenate the parts
        y = np.concatenate((fft_out_r[: N//2, :], fft_out_r[start_idx_second_part:end_idx_second_part, :]))
        hpa_y = np.concatenate((fft_out_hpa[: N//2, :], fft_out_hpa[start_idx_second_part:end_idx_second_part, :]))
        hpa_y_dvae = np.concatenate((fft_out_hpa_dvae[: N//2, :], fft_out_hpa_dvae[start_idx_second_part:end_idx_second_part, :])) # DVAE
        rx_ps_r = y.reshape(1, -1)
        rx_ps_hpa = hpa_y.reshape(1, -1)
        rx_ps_hpa_dvae = hpa_y_dvae.reshape(1, -1) # DVAE

        # Calculate BER for the current channel type
        _, current_errb_array[m] = decision(D, M, b.flatten(), rx_ps_r.flatten())
        _, hpa_errb_array[m] = decision(D, M, b.flatten(), rx_ps_hpa.flatten())
        _, dvae_errb_array[m] = decision(D, M, b.flatten(), rx_ps_hpa_dvae.flatten()) # DVAE

    simulated_bers[f'{channel_name} (P-CSI)'] = current_errb_array # Update label
    hpa_bers[f'{channel_name} (P-CSI)'] = hpa_errb_array
    dvae_bers[f'{channel_name} (P-CSI)'] = dvae_errb_array # DVAE

    print(f"BER for {channel_name} Fading with P-CSI: {current_errb_array}\n")
    print(f"HPA BER for {channel_name} Fading with P-CSI: {hpa_errb_array}\n")
    print(f"DVAE BER for {channel_name} Fading with P-CSI: {dvae_errb_array}\n") # DVAE

# Calculate theoretical BER for AWGN and Rayleigh
for i in range(len(SNR)):
    t_snr_linear = 10 ** (SNR[i] / 10) # Es/N0
    eb_n0_linear = t_snr_linear / np.log2(M) # Eb/N0 for QPSK

    # Theoretical BER for QPSK in AWGN
    theo_ber_awgn[i] = (1 / 2) * erfc(np.sqrt(t_snr_linear)) - (1 / 8) * (erfc(np.sqrt(t_snr_linear))) ** 2
    # Theoretical BER for QPSK in Rayleigh fading with Coherent Detection (P-CSI)
    # Pbe = 0.5 * (1 - sqrt(Eb/N0 / (1 + Eb/N0)))
    # theo_ber_rayleigh[i] = 0.5 * (1 - np.sqrt(eb_n0_linear / (1 + eb_n0_linear)))

plt.figure(4)
plt.semilogy(SNR, theo_ber_awgn, 'k', label='Theoretical BER (AWGN)')
#plt.semilogy(SNR, theo_ber_rayleigh, 'g-.', label='Theoretical BER (Rayleigh P-CSI)')

# Plot simulated BERs
if 'Rayleigh (P-CSI)' in simulated_bers:
    plt.semilogy(SNR, simulated_bers['Rayleigh (P-CSI)'], 'b--', label='Simulated BER (Rayleigh P-CSI)')
    plt.semilogy(SNR, hpa_bers['Rayleigh (P-CSI)'], 'm-', label='Simulated HPA BER (Rayleigh P-CSI)')
    plt.semilogy(SNR, dvae_bers['Rayleigh (P-CSI)'], 'r-.', label='Simulated DVAE BER (Rayleigh P-CSI)') # DVAE
if 'Rician (P-CSI)' in simulated_bers:
    plt.semilogy(SNR, simulated_bers['Rician (P-CSI)'], 'b--', label='Simulated BER (Rician K=4 P-CSI)')
    plt.semilogy(SNR, hpa_bers['Rician (P-CSI)'], 'm-', label='Simulated HPA BER (Rician K=4 P-CSI)')
    plt.semilogy(SNR, dvae_bers['Rician (P-CSI)'], 'r-.', label='Simulated DVAE BER (Rician K=4 P-CSI)') # DVAE

plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.axis([0, 18, 1e-5, 1]) # Adjusted x-axis to cover full SNR range and y-axis for better visibility
plt.legend()
plt.grid(True, which="both", ls="-") # Add grid for better readability
plt.title('BER vs SNR for Different Fading Channels with Perfect CSI')
plt.savefig('BER_plot_fading_with_csi.png')

# EVM 결과 재출력 (스크롤 없이 마지막에 확인용)
print("\n[최종 EVM 요약]")
print(f"  EVM (보상 없음, HPA):  {evm_hpa:.4f}%")
print(f"  EVM (DVAE 보상):       {evm_dvae:.4f}%")
