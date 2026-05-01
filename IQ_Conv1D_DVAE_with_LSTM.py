import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import re

# dvae.py에서 DVAE, Sampling, polynomial_tf import
# (NonlinearMemory/dvae.py 경로에 위치)
module_path = '/content/drive/MyDrive'
if module_path not in sys.path:
    sys.path.append(module_path)
from NonlinearMemory.dvae import Sampling, DVAE, polynomial_tf

# -----------------------------
# RMS Normalization
# -----------------------------
def normalize_with_rms(I_data, Q_data):

    magnitude = np.sqrt(I_data**2 + Q_data**2)

    rms = np.sqrt(np.mean(magnitude**2))

    if rms > 0:
        return I_data/rms, Q_data/rms, rms

    return I_data, Q_data, 1.0


# -----------------------------
# complex -> 2 channel
# -----------------------------
def to_2ch(x):

    return np.stack([np.real(x), np.imag(x)], axis=-1)

def to_1ch(i, q):
    return np.stack([i, q], axis=-1).astype(np.float32)

# convert to complex
def to_complex(x2ch):
    return x2ch[...,0] + 1j * x2ch[...,1]


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


# -----------------------------
# Model Build
# -----------------------------
N = 64
seq_len = N
beta_kl = 1e-3  # KL weight (tune)
back_off = 5    # HPA backoff (dB) — polynomial_tf와 동일하게 유지
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

# DLA: 원본 신호(outputData)를 기준으로 정규화
# 훈련/추론 모두 원본 신호가 입력이므로 정규화 기준이 자동으로 일치
_, _, rms_train = normalize_with_rms(outputData_train_I, outputData_train_Q)

# 원본 신호 정규화
target_I_train = outputData_train_I / rms_train
target_Q_train = outputData_train_Q / rms_train
target_I_val   = outputData_val_I   / rms_train
target_Q_val   = outputData_val_Q   / rms_train
target_I_test  = outputData_test_I  / rms_train
target_Q_test  = outputData_test_Q  / rms_train

print(f"rms_train (원본 신호 기준): {rms_train:.6f}")
print(f"Shape of target_I_train after split and norm: {target_I_train.shape}")

# DLA: X_in = X_out = 원본 신호
# loss는 train_step 내부에서 HPA 통과 후 계산 → 별도 target 불필요
X_in_train_ch  = to_1ch(target_I_train, target_Q_train)
X_out_train_ch = to_1ch(target_I_train, target_Q_train)   # X_in과 동일
X_in_val_ch    = to_1ch(target_I_val,   target_Q_val)
X_out_val_ch   = to_1ch(target_I_val,   target_Q_val)
X_in_test_ch   = to_1ch(target_I_test,  target_Q_test)
X_out_test_ch  = to_1ch(target_I_test,  target_Q_test)

print("Shapes of final training data (X_out_train_ch, X_in_train_ch) *before* fit:")
print(f"X_out_train_ch.shape: {X_out_train_ch.shape}, X_in_train_ch.shape: {X_in_train_ch.shape}")

encoder = build_encoder(seq_len)

decoder = build_decoder(seq_len)

dvae = DVAE(encoder, decoder, beta=beta_kl, backoff=float(back_off))

dvae.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3)
)

dvae.build((None,seq_len,2))

dvae.summary()


# ---------------------------------
# Example Training Data
# ---------------------------------
# --- MODIFIED PART FOR RESUMING TRAINING ---
model_dir = '/content/drive/MyDrive/NonlinearMemory/'
saved_models = [f for f in os.listdir(model_dir) if f.startswith('my_dvae_model_') and f.endswith('.weights.h5')]
latest_epoch = 0
latest_model_path = None

if saved_models:
    epoch_numbers = []
    for model_file in saved_models:
        match = re.search(r'my_dvae_model_(\d+)\.weights.h5', model_file)
        if match:
            epoch_numbers.append(int(match.group(1)))

    if epoch_numbers:
        latest_epoch = max(epoch_numbers)
        latest_model_path = os.path.join(model_dir, f'my_dvae_model_{latest_epoch}.weights.h5')

        print(f"Latest saved model found: {latest_model_path}. Loading model to resume training.")
        # Ensure DVAE and Sampling classes are available for custom_objects
        # dvae = models.load_model(latest_model_path, custom_objects={'DVAE': DVAE, 'Sampling': Sampling})
        # dvae.compile(optimizer=tf.keras.optimizers.Adam(1e-3)) # Recompile after loading
        dvae.load_weights(latest_model_path)
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

    # -----------------------------
    # Save model
    # -----------------------------
    # 기존 코드
    # model_save_path_keras = os.path.join(model_dir, f'my_dvae_model_{target_save_epoch_total}.keras')
    # dvae.save(model_save_path_keras)

    # 변경된 코드
    model_save_path_weights = os.path.join(model_dir, f'my_dvae_model_{target_save_epoch_total}.weights.h5')
    dvae.save_weights(model_save_path_weights)
    print(f'DVAE 가중치가 {model_save_path_weights}에 저장되었습니다.')

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