# IQ_Conv1D_DVAE_with_LSTM.py
# Keras/TensorFlow implementation of Conv1D + LSTM DVAE for complex IQ signals,
# with NMSE, EVM, ACPR evaluation and example synthetic HPA dataset.
import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import re # Added for regular expressions
import importlib.util # Added for robust module import

# Define the path where dvae.py is located
module_path = '/content/drive/MyDrive/NonlinearMemory'
module_name = 'dvae'
file_path = os.path.join(module_path, f'{module_name}.py')

# Create a module spec from the file path
spec = importlib.util.spec_from_file_location(module_name, file_path)
if spec is None:
    raise ModuleNotFoundError(f"Cannot find {module_name}.py at {file_path}")

# Load the module
dvae_module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = dvae_module
spec.loader.exec_module(dvae_module)

# Import Sampling and DVAE directly from the loaded module
Sampling = dvae_module.Sampling
DVAE = dvae_module.DVAE

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

# -------------------------
# 1) Utilities: metrics
# -------------------------
def compute_nmse_db(pred_complex, ref_complex):
    num = np.mean(np.abs(pred_complex - ref_complex)**2)
    den = np.mean(np.abs(ref_complex)**2) + 1e-12
    nmse = num / den
    return 10 * np.log10(nmse + 1e-12)

def compute_evm_percent(pred_complex, ref_complex):
    # EVM = sqrt(sum |ref - pred|^2 / sum |ref|^2) * 100%
    num = np.sum(np.abs(ref_complex - pred_complex)**2)
    den = np.sum(np.abs(ref_complex)**2) + 1e-12
    evm = np.sqrt(num / den) * 100.0
    return evm

def compute_acpr_db(sig_complex, fs, signal_bw):
    """
    Simple ACPR estimate for baseband signal:
      - compute PSD (FFT) of signal
      - define main band as [-signal_bw/2, +signal_bw/2]
      - adjacent bands: [-3/2*BW, -1/2*BW] and [1/2*BW, 3/2*BW]
    Returns ratio (adjacent power / main power) in dB for the worst adjacent.
    Note: works for single-channel baseband; for real RF you'd upconvert and measure.
    """
    N = len(sig_complex)
    # compute FFT freq axis
    S = np.fft.fftshift(np.fft.fft(sig_complex * np.hanning(N)))
    psd = np.abs(S)**2
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1/fs))  # Hz

    bw = signal_bw
    half = bw / 2.0

    # indices
    main_idx = np.where((freqs >= -half) & (freqs <= half))[0]
    left_idx = np.where((freqs >= -1.5*bw) & (freqs < -0.5*bw))[0]
    right_idx = np.where((freqs > 0.5*bw) & (freqs <= 1.5*bw))[0]

    main_power = np.sum(psd[main_idx]) + 1e-12
    left_power = np.sum(psd[left_idx]) + 1e-12
    right_power = np.sum(psd[right_idx]) + 1e-12

    acpr_left = 10*np.log10(left_power / main_power)
    acpr_right = 10*np.log10(right_power / main_power)

    return min(acpr_left, acpr_right)  # return worst (lowest dB -> more leakage)

# Encoder
def build_encoder(seq_len, channels=2, latent_dim=64):
    inp = layers.Input(shape=(seq_len, channels), name="encoder_input")
    x = layers.Conv1D(64, 7, strides=2, padding='same', activation='relu')(inp)   # /2
    x = layers.Conv1D(128, 5, strides=2, padding='same', activation='relu')(x)    # /4
    x = layers.Conv1D(256, 3, strides=2, padding='same', activation='relu')(x)    # /8
    # optional batchnorm
    x = layers.LayerNormalization()(x)
    # LSTM to capture long-term dependencies (return last)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=False))(x)  # collapse time
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    encoder = models.Model(inp, [z_mean, z_log_var], name="encoder")
    return encoder

# Decoder
def build_decoder(seq_len, channels=2, latent_dim=64):
    latent_in = layers.Input(shape=(latent_dim,), name="z_input")
    x = layers.Dense((seq_len // 8) * 256, activation='relu')(latent_in)
    x = layers.Reshape((seq_len // 8, 256))(x)
    x = layers.UpSampling1D(2)(x)  # /4
    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.UpSampling1D(2)(x)  # /2
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = layers.UpSampling1D(2)(x)  # /1
    x = layers.Conv1D(32, 3, padding='same', activation='relu')(x)
    out = layers.Conv1D(channels, 3, padding='same', activation='linear')(x)
    decoder = models.Model(latent_in, out, name="decoder")
    return decoder

# Convert to real/imag 2-channel for model: shape (N, seq_len, 2)
def to_2ch(z):
    return np.stack([np.real(z), np.imag(z)], axis=-1).astype(np.float32)

def to_1ch(i, q):
    return np.stack([i, q], axis=-1).astype(np.float32)

# convert to complex
def to_complex(x2ch):
    return x2ch[...,0] + 1j * x2ch[...,1]

tf.random.set_seed(0)
np.random.seed(0)

N = 64
seq_len = N  # sequence length (time samples) # 각 입력 시퀀스의 길이 (시간 샘플 수)
fs = 1.0        # normalized sample rate (Hz) - used for ACPR relative scale
signal_bw = 0.2 # normalized bandwidth fraction for ACPR calc
# -------------------------
# 4) Model: Conv1D + LSTM DVAE
# -------------------------
latent_dim = N * 2  # 잠재 공간의 차원
beta_kl = 1e-6  # KL weight (tune)
batch_size = N * 2 # 훈련 시 한 번에 처리되는 샘플의 수

file1_I	 = '/content/drive/MyDrive/input_iq_I.csv'
file1_Q	 = '/content/drive/MyDrive/input_iq_Q.csv'

file2_I	 = '/content/drive/MyDrive/input_target_I.csv'
file2_Q	 = '/content/drive/MyDrive/input_target_Q.csv'

inputData_I = pd.read_csv(file1_I,header=None)
inputData_Q = pd.read_csv(file1_Q,header=None)
outputData_I = pd.read_csv(file2_I,header=None)
outputData_Q = pd.read_csv(file2_Q,header=None)
print(f"Initial outputData_I shape (from CSV): {outputData_I.shape}")
print(f"Initial outputData_Q shape (from CSV): {outputData_Q.shape}")

print("Complete import data")
inputData_I = inputData_I.to_numpy()
inputData_Q = inputData_Q.to_numpy()
outputData_I = outputData_I.to_numpy()
outputData_Q = outputData_Q.to_numpy()

print(f"Shape of inputData_I after to_numpy(): {inputData_I.shape}")
print(f"Shape of outputData_I after to_numpy(): {outputData_I.shape}")

# Reshape data to (num_sequences, seq_len)
# Assuming original numpy arrays are (total_samples, 1)
num_total_samples_I = inputData_I.shape[0]
if num_total_samples_I % seq_len != 0:
    raise ValueError(f"Total samples for I ({num_total_samples_I}) must be divisible by seq_len ({seq_len}) for reshaping.")

num_total_samples_Q = inputData_Q.shape[0]
if num_total_samples_Q % seq_len != 0:
    raise ValueError(f"Total samples for Q ({num_total_samples_Q}) must be divisible by seq_len ({seq_len}) for reshaping.")

inputData_I = inputData_I.reshape(-1, seq_len)
inputData_Q = inputData_Q.reshape(-1, seq_len)
outputData_I = outputData_I.reshape(-1, seq_len)
outputData_Q = outputData_Q.reshape(-1, seq_len)

# Now, len_i, len_q, etc. should represent the number of *sequences*
len_i = inputData_I.shape[0] # Number of sequences
len_q = inputData_Q.shape[0]
len_o_I = outputData_I.shape[0]
len_o_Q = outputData_Q.shape[0]

print("----------------------------")
print(f"Shape of inputData_I after reshape: {inputData_I.shape}, Number of sequences (len_i): {len_i}")
print(f"Shape of inputData_Q after reshape: {inputData_Q.shape}, Number of sequences (len_q): {len_q}")
print(f"Shape of outputData_I after reshape: {outputData_I.shape}, Number of sequences (len_o_I): {len_o_I}")
print(f"Shape of outputData_Q after reshape: {outputData_Q.shape}, Number of sequences (len_o_Q): {len_o_Q}")
print("----------------------------")

# train/test split - ensure n_train and n_val are based on number of sequences
n_val = int(0.1 * len_i)
n_train = int(0.8 * len_i) # Define n_train for consistent splitting

inputData_I_train = inputData_I[0:n_train]
inputData_I_val = inputData_I[n_train: n_train + n_val]
inputData_I_test = inputData_I[n_train + n_val: ]
inputData_Q_train = inputData_Q[0:n_train]
inputData_Q_val = inputData_Q[n_train: n_train + n_val]
inputData_Q_test = inputData_Q[n_train + n_val: ]

outputData_train_I = outputData_I[0:n_train]
outputData_train_Q = outputData_Q[0:n_train]
outputData_val_I = outputData_I[n_train: n_train + n_val]
outputData_val_Q = outputData_Q[n_train: n_train + n_val]
outputData_test_I = outputData_I[n_train + n_val: ]
outputData_test_Q = outputData_Q[n_train + n_val: ]

# Capture the third return value (rms_magnitude)
inputData_I_train, inputData_Q_train, _ = normalize_with_rms(inputData_I_train, inputData_Q_train)
inputData_I_val, inputData_Q_val, _ = normalize_with_rms(inputData_I_val, inputData_Q_val)
inputData_I_test, inputData_Q_test, _ = normalize_with_rms(inputData_I_test, inputData_Q_test)
target_I_train, target_Q_train, _ = normalize_with_rms(outputData_train_I, outputData_train_Q)
target_I_val, target_Q_val, _ = normalize_with_rms(outputData_val_I, outputData_val_Q)
target_I_test, target_Q_test, _ = normalize_with_rms(outputData_test_I, outputData_test_Q)

print(f"Shape of inputData_I_train after split and norm: {inputData_I_train.shape}")
print(f"Shape of target_I_train after split and norm: {target_I_train.shape}")

X_in_train_ch = to_1ch(inputData_I_train, inputData_Q_train)
X_out_train_ch = to_1ch(target_I_train, target_Q_train)
X_in_val_ch = to_1ch(inputData_I_val, inputData_Q_val)
X_out_val_ch = to_1ch(target_I_val, target_Q_val)
X_in_test_ch = to_1ch(inputData_I_test, inputData_Q_test)
X_out_test_ch = to_1ch(target_I_test, target_Q_test)

print("Shapes of final training data (X_out_train_ch, X_in_train_ch) *before* fit:")
print(f"X_out_train_ch.shape: {X_out_train_ch.shape}, X_in_train_ch.shape: {X_in_train_ch.shape}")

encoder = build_encoder(seq_len, channels=2, latent_dim=latent_dim)
decoder = build_decoder(seq_len, channels=2, latent_dim=latent_dim)

# Full VAE model
enc_in = layers.Input(shape=(seq_len, 2), name="vae_input")       # HPA output (distorted)
z_mean, z_log_var = encoder(enc_in)
z = Sampling()([z_mean, z_log_var])
dec_out = decoder(z)   # predicted original input (real/imag 2ch)
vae = models.Model(enc_in, dec_out, name="IQ_Conv1D_LSTM_DVAE")

# Losses: reconstruction + KL
# The original line caused a ValueError because it misused KerasTensors.
# The actual loss calculation is handled within the DVAE class's train_step.
# recon_loss = tf.reduce_mean(tf.square(tf.cast(dec_out, tf.float32) - tf.cast(layers.Input(shape=(seq_len,2)), tf.float32)))
# The above cannot refer to target directly; instead, in training we'll compute loss via custom train_step
# Simpler: implement custom train_step by subclassing Model for accurate recon + kl

# instantiate
dvae = DVAE(encoder, decoder, beta=beta_kl)
dvae.compile(optimizer=tf.keras.optimizers.Adam(1e-3))

# --- MODIFIED PART FOR RESUMING TRAINING ---
model_dir = '/content/drive/MyDrive/NonlinearMemory/'
saved_models = [f for f in os.listdir(model_dir) if f.startswith('my_dvae_model_') and f.endswith('.keras')]
latest_epoch = 0
latest_model_path = None

if saved_models:
    epoch_numbers = []
    for model_file in saved_models:
        match = re.search(r'my_dvae_model_(\d+)\.keras', model_file)
        if match:
            epoch_numbers.append(int(match.group(1)))

    if epoch_numbers:
        latest_epoch = max(epoch_numbers)
        latest_model_path = os.path.join(model_dir, f'my_dvae_model_{latest_epoch}.keras')

        print(f"Latest saved model found: {latest_model_path}. Loading model to resume training.")
        # Ensure DVAE and Sampling classes are available for custom_objects
        dvae = models.load_model(latest_model_path, custom_objects={'DVAE': DVAE, 'Sampling': Sampling})
        dvae.compile(optimizer=tf.keras.optimizers.Adam(1e-3)) # Recompile after loading
        print(f"Resuming training from total epochs: {latest_epoch}.")
    else:
        print("No parsable DVAE models found in the directory. Starting new training.")
else:
    print("No saved DVAE models found. Starting new training.")

# -------------------------
# 5) Training
# epoch_num defines how many epochs to run in each segment (e.g., [5, 5, ...])
# epoch_num2 defines the *cumulative* epoch number for saving (e.g., [5, 10, ...])

epoch_num = [5] * 20
epoch_num2 = [i * 5 for i in range(1, 21)]

# Adjust the loop to start from the correct point if resuming
start_idx = 0
if latest_epoch > 0:
    try:
        # Find the index in epoch_num2 that corresponds to the latest_epoch
        # We need to start training *after* this epoch has completed.
        # So, if latest_epoch is 15, and epoch_num2 is [5, 10, 15, 20, ...],
        # we want to start from the index *after* 15 (which is for 20).
        start_idx = epoch_num2.index(latest_epoch)
        # If the latest_epoch was exactly one of the target_save_epoch_total, then we continue from the next segment.
        if start_idx < len(epoch_num2) - 1 and epoch_num2[start_idx] == latest_epoch:
            start_idx += 1
        print(f"Training will continue from the segment after {latest_epoch} total epochs.")
    except ValueError:
        print(f"Warning: Latest epoch {latest_epoch} not found in saving schedule (epoch_num2). Starting from the beginning of the schedule.")
        start_idx = 0

for idx in range(start_idx, len(epoch_num)):
  current_epochs_to_run = epoch_num[idx]
  target_save_epoch_total = epoch_num2[idx]

  print(f"\n--- Training for {current_epochs_to_run} epochs (cumulative total: {target_save_epoch_total} epochs) ---")
  history = dvae.fit(
    x=X_in_train_ch,
    y=X_out_train_ch,
    validation_data=(X_in_val_ch, X_out_val_ch),
    epochs=current_epochs_to_run,
    batch_size=batch_size
  )

  model_save_path_keras = os.path.join(model_dir, f'my_dvae_model_{target_save_epoch_total}.keras')
  dvae.save(model_save_path_keras) # 확장자를 .keras로 변경
  print(f'DVAE 모델이 {model_save_path_keras}에 저장되었습니다.')
  print("evaluate shape: ", X_in_test_ch.shape, X_out_test_ch.shape)
  dvae.evaluate(X_in_test_ch, X_out_test_ch)

  # The original code loaded the just-saved model into `loaded_dvae_keras` but did not use it for subsequent training.
  # Removing the redundant load as `dvae` is already the live, updated model.
  # If a restart happens, the initial logic at the top of the cell will handle loading the latest model.

# --- END MODIFIED PART ---

# -------------------------
# 6) Evaluation on test set
# -------------------------
# predict restored inputs from HPA outputs
pred_test = dvae.predict(X_in_test_ch, batch_size=batch_size)

pred_test_c = to_complex(pred_test)
hpa_test_c = to_complex(X_in_test_ch)
ref_test_c = to_complex(X_out_test_ch)

# compute NMSE/EVM/ACPR per sample and average
nmse_list = []
evm_list = []
acpr_list = []
for i in range(len(pred_test_c)):
    nmse_list.append(compute_nmse_db(pred_test_c[i], ref_test_c[i]))
    evm_list.append(compute_evm_percent(pred_test_c[i], ref_test_c[i]))
    acpr_list.append(compute_acpr_db(pred_test_c[i], fs=fs, signal_bw=signal_bw))

print("Test NMSE (dB) mean: {:.2f}".format(np.mean(nmse_list)))
print("Test EVM (%) mean: {:.2f}".format(np.mean(evm_list)))
print("Test ACPR (dB, worst adjacent) mean: {:.2f}".format(np.mean(acpr_list)))

# also compute baseline (no pred, just identity mapping: pass HPA output through simple inverse?)
# For baseline, measure how bad HPA output vs original input is (i.e., without predistortion)
nmse_out_list = [compute_nmse_db(hpa_test_c[i], ref_test_c[i]) for i in range(len(hpa_test_c))]
evm_out_list = [compute_evm_percent(hpa_test_c[i], ref_test_c[i]) for i in range(len(hpa_test_c))]
acpr_out_list = [compute_acpr_db(hpa_test_c[i], fs=fs, signal_bw=signal_bw) for i in range(len(hpa_test_c))]

print("Baseline (HPA output -> original) NMSE (dB) mean: {:.2f}".format(np.mean(nmse_out_list)))
print("Baseline EVM (%) mean: {:.2f}".format(np.mean(evm_out_list)))
print("Baseline ACPR (dB) mean: {:.2f}".format(np.mean(acpr_out_list)))

# -------------------------
# 7) Visualize one example: waveforms and spectra
# -------------------------
idx = 5
pred_c = pred_test_c[idx]
ref_c = ref_test_c[idx]
out_c = hpa_test_c[idx]
t = np.arange(seq_len)

plt.figure(figsize=(12,6))
plt.subplot(2,1,1)
plt.plot(t, np.real(out_c), label="HPA output (input to model)", alpha=0.6)
plt.plot(t, np.real(ref_c), label="Target original input", alpha=0.6)
plt.plot(t, np.real(pred_c), label="DVAE restored input", alpha=0.8)
plt.title("Real part (time domain)")
plt.legend()

plt.subplot(2,1,2)
# PSDs
def plot_psd(sig, label):
    S = np.fft.fftshift(np.fft.fft(sig * np.hanning(len(sig))))
    psd = 20*np.log10(np.abs(S)+1e-12)
    freqs = np.fft.fftshift(np.fft.fftfreq(len(sig), d=1/fs))
    plt.plot(freqs, psd, label=label)

plot_psd(out_c, "HPA output")
plot_psd(ref_c, "Target")
plot_psd(pred_c, "Restored")
plt.title("PSD (dB)")
plt.xlim(-0.5, 0.5)
plt.legend()
plt.tight_layout()
plt.show()