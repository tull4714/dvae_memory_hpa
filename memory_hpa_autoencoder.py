import numpy as np
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

# Add the parent directory of NonlinearMemory to sys.path
module_path = '/content/drive/MyDrive'
if module_path not in sys.path:
    sys.path.append(module_path)

# Explicitly add the NonlinearMemory directory to sys.path and import dvae directly
# This resolves the ModuleNotFoundError encountered previously.
nonlinear_memory_path = '/content/drive/MyDrive/NonlinearMemory'
if nonlinear_memory_path not in sys.path:
    sys.path.insert(0, nonlinear_memory_path)

# Clear any cached references to 'dvae' or 'NonlinearMemory' before re-importing
if 'dvae' in sys.modules:
    del sys.modules['dvae']
if 'NonlinearMemory' in sys.modules:
    del sys.modules['NonlinearMemory']
for module_name in list(sys.modules.keys()):
    if module_name.startswith('NonlinearMemory.'):
        del sys.modules[module_name]

# Now import dvae directly as a module, and then extract Sampling and DVAE
import dvae
from dvae import Sampling
from dvae import DVAE

def polynomial(p_in, backoff):
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
    """

    numoferror = 0
    numoferror_b = 0

    if M > 1:
        log2M = np.log2(M)
    else:
        log2M = 1

    for k in range(D):

        r_out = np.real(signal_out[k])
        i_out = np.imag(signal_out[k])

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

        # --- 4. 심볼 오류 계산 (Pse) ---
        if decis != signal_in[k]:
            numoferror += 1

        # --- 5. 비트 오류 계산 (Pbe) ---
        r_in = np.real(signal_in[k])
        i_in = np.imag(signal_in[k])

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
        s = (-2 * np.fix(rand_I * 4) + 3) + 1j * (-2 * np.fix(rand_Q * 4) + 3)

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

    return ber

# -----------------------------
# Encoder
# -----------------------------
def build_encoder(seq_len):

    inputs = layers.Input(shape=(seq_len,2))

    x = layers.Conv1D(64,5,padding="same",activation="relu")(inputs)

    x = layers.Conv1D(64,5,padding="same",activation="relu")(x)

    x = layers.Conv1D(128,3,padding="same",activation="relu")(x)

    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=False)
    )(x)

    z_mean = layers.Dense(64)(x)

    z_log_var = layers.Dense(64)(x)

    return models.Model(inputs,[z_mean,z_log_var],name="encoder")


# -----------------------------
# Decoder
# -----------------------------
def build_decoder(seq_len, channels=2, latent_dim=64):

    latent_inputs = layers.Input(shape=(latent_dim,))

    x = layers.Dense(seq_len*128,activation="relu")(latent_inputs)

    x = layers.Reshape((seq_len,128))(x)

    x = layers.Conv1D(128,5,padding="same",activation="relu")(x)

    x = layers.Conv1D(64,5,padding="same",activation="relu")(x)

    outputs = layers.Conv1D(channels,1,padding="same")(x)

    return models.Model(latent_inputs,outputs,name="decoder")

Fs = 4          # frequency of the sample
Fd = 1          # frequency of the data
N = 64
seq_len = N
beta_kl = 1e-3
back_off = 5
Block = 100000
M = 4
D = N * Block

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
keras.backend.clear_session()
name = '/content/drive/MyDrive/NonlinearMemory/my_dvae_model_100.weights.h5'

# 모델 구조 재생성 후 가중치 로드
encoder = build_encoder(seq_len)
decoder = build_decoder(seq_len)
dvae = DVAE(encoder, decoder, beta=beta_kl, backoff=float(back_off))
dvae.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
dvae.build((None, seq_len, 2))
dvae.load_weights(name)
print(f'DVAE 가중치를 {name}에서 성공적으로 로드했습니다.')

# DLA 추론: 원본 신호 → predistorted 신호
# predict() 대신 직접 call() 사용 — outputs만 명시적으로 추출
sig_in_Linear_list = []
for i in range(0, len(dvae_1ch), N):
    batch = tf.constant(dvae_1ch[i:i+N])
    predistorted, _, _ = dvae(batch, training=False)   # outputs만 추출
    sig_in_Linear_list.append(predistorted.numpy())
sig_in_Linear = np.concatenate(sig_in_Linear_list, axis=0)

sig_in_Linear_c = to_complex(sig_in_Linear)
sp_dvae_norm = sig_in_Linear_c.reshape(-1, 1)

# 역정규화 (원본 신호 RMS 기준)
dvae_i_r, dvae_q_r = denormalize_with_rms(np.real(sp_dvae_norm).flatten(),
                                            np.imag(sp_dvae_norm).flatten(),
                                            rms_mag)
dvae_r = (dvae_i_r + 1j * dvae_q_r).reshape(-1, 1)

hpa_data_tx, _, _ = polynomial(IQ_out_r, back_off)
hpa_dvae_r, _, _ = polynomial(dvae_r, back_off)     # DVAE

sigpwr = np.linalg.norm(IQ_out_r) ** 2 / len(IQ_out_r)

# Define channel k-factor values: 0 for Rayleigh, 4 for Rician
channel_k_values = {0: 'Rayleigh', 4: 'Rician'}
simulated_bers = {} # Dictionary to store BERs for each K-factor
hpa_bers = {}

# --- DVAE BER 결과를 저장할 변수 초기화 ---
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
