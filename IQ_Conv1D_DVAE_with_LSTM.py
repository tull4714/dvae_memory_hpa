# IQ_Conv1D_DPD_Deterministic.py
# Deterministic Conv1D + LSTM DPD for IQ complex signals

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import re # Added for regular expressions

# -------------------------
# 1) Utilities
# -------------------------

def normalize_with_rms(I_data, Q_data):
    magnitude = np.sqrt(I_data**2 + Q_data**2)
    rms = np.sqrt(np.mean(magnitude**2))
    if rms > 0:
        return I_data/rms, Q_data/rms, rms
    return I_data, Q_data, 1.0

def to_2ch(i, q):
    return np.stack([i, q], axis=-1).astype(np.float32)

def to_complex(x2):
    return x2[...,0] + 1j*x2[...,1]

def compute_nmse_db(pred, ref):
    num = np.mean(np.abs(pred-ref)**2)
    den = np.mean(np.abs(ref)**2)+1e-12
    return 10*np.log10(num/den+1e-12)

# -------------------------
# 2) Model: Deterministic DPD
# -------------------------

def build_dpd_model(seq_len):

    inp = layers.Input(shape=(seq_len,2))

    # time-aligned mapping (no stride!)
    x = layers.Conv1D(128, 5, padding="same", activation="relu")(inp)
    x = layers.Conv1D(128, 5, padding="same", activation="relu")(x)

    x = layers.LSTM(128, return_sequences=True)(x)

    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    correction = layers.Conv1D(2, 1, padding="same")(x)

    # residual connection (VERY IMPORTANT)
    out = layers.Add()([inp, correction])

    return models.Model(inp, out, name="Deterministic_DPD")

# -------------------------
# 3) Data Loading
# -------------------------

seq_len = 64
batch_size = 64

file1_I = '/content/drive/MyDrive/input_iq_I.csv'
file1_Q = '/content/drive/MyDrive/input_iq_Q.csv'
file2_I = '/content/drive/MyDrive/input_target_I.csv'
file2_Q = '/content/drive/MyDrive/input_target_Q.csv'

input_I = pd.read_csv(file1_I,header=None).to_numpy()
input_Q = pd.read_csv(file1_Q,header=None).to_numpy()
target_I = pd.read_csv(file2_I,header=None).to_numpy()
target_Q = pd.read_csv(file2_Q,header=None).to_numpy()

# Reshape data to (num_sequences, seq_len)
# Assuming original numpy arrays are (total_samples, 1)
num_total_samples_I = input_I.shape[0]
if num_total_samples_I % seq_len != 0:
    raise ValueError(f"Total samples for I ({num_total_samples_I}) must be divisible by seq_len ({seq_len}) for reshaping.")

num_total_samples_Q = input_Q.shape[0]
if num_total_samples_Q % seq_len != 0:
    raise ValueError(f"Total samples for Q ({num_total_samples_Q}) must be divisible by seq_len ({seq_len}) for reshaping.")

# reshape
input_I = input_I.reshape(-1, seq_len)
input_Q = input_Q.reshape(-1, seq_len)
target_I = target_I.reshape(-1, seq_len)
target_Q = target_Q.reshape(-1, seq_len)

n_total = input_I.shape[0]
n_train = int(0.8*n_total)
n_val = int(0.1*n_total)

# split
I_train = input_I[:n_train]
Q_train = input_Q[:n_train]
I_val = input_I[n_train:n_train+n_val]
Q_val = input_Q[n_train:n_train+n_val]
I_test = input_I[n_train+n_val:]
Q_test = input_Q[n_train+n_val:]

tI_train = target_I[:n_train]
tQ_train = target_Q[:n_train]
tI_val = target_I[n_train:n_train+n_val]
tQ_val = target_Q[n_train:n_train+n_val]
tI_test = target_I[n_train+n_val:]
tQ_test = target_Q[n_train+n_val:]

# normalize
I_train,Q_train,_ = normalize_with_rms(I_train,Q_train)
I_val,Q_val,_ = normalize_with_rms(I_val,Q_val)
I_test,Q_test,_ = normalize_with_rms(I_test,Q_test)

tI_train,tQ_train,_ = normalize_with_rms(tI_train,tQ_train)
tI_val,tQ_val,_ = normalize_with_rms(tI_val,tQ_val)
tI_test,tQ_test,_ = normalize_with_rms(tI_test,tQ_test)

X_train = to_2ch(I_train,Q_train)
Y_train = to_2ch(tI_train,tQ_train)

X_val = to_2ch(I_val,Q_val)
Y_val = to_2ch(tI_val,tQ_val)

X_test = to_2ch(I_test,Q_test)
Y_test = to_2ch(tI_test,tQ_test)

# -------------------------
# 4) Train
# -------------------------

model = build_dpd_model(seq_len)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='mse'
)

model.summary()

# --- MODIFIED PART FOR RESUMING TRAINING ---
model_dir = '/content/drive/MyDrive/NonlinearMemory/'
saved_models = [f for f in os.listdir(model_dir) if f.startswith('my_dpd_model_') and f.endswith('.keras')] # Changed prefix
latest_epoch = 0
latest_model_path = None

if saved_models:
    epoch_numbers = []
    for model_file in saved_models:
        match = re.search(r'my_dpd_model_(\d+)\.keras', model_file) # Changed regex
        if match:
            epoch_numbers.append(int(match.group(1)))

    if epoch_numbers:
        latest_epoch = max(epoch_numbers)
        latest_model_path = os.path.join(model_dir, f'my_dpd_model_{latest_epoch}.keras') # Changed filename

        print(f"Latest saved DPD model found: {latest_model_path}. Loading model to resume training.") # Updated print
        # custom_objects are not needed for this standard Keras DPD model
        model = models.load_model(latest_model_path) # Removed custom_objects
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mse') # Recompile after loading (with DPD specific settings)
        print(f"Resuming training DPD model from total epochs: {latest_epoch}.") # Updated print
    else:
        print("No parsable DPD models found in the directory. Starting new training.") # Updated print
else:
    print("No saved DPD models found. Starting new training.") # Updated print

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

  history = model.fit(
      X_train,
      Y_train,
      validation_data=(X_val,Y_val),
      epochs=current_epochs_to_run,
      batch_size=batch_size
  )

  model_save_path_keras = os.path.join(model_dir, f'my_dpd_model_{target_save_epoch_total}.keras') # Changed filename
  model.save(model_save_path_keras) # 확장자를 .keras로 변경
  print(f'DPD 모델이 {model_save_path_keras}에 저장되었습니다.') # Updated print
  print("evaluate shape: ", X_test.shape, Y_test.shape)
  model.evaluate(X_test, Y_test)

# -------------------------
# 5) Evaluation
# -------------------------

pred = model.predict(X_test)

pred_c = to_complex(pred)
ref_c = to_complex(Y_test)
in_c = to_complex(X_test)

nmse_pred = np.mean([compute_nmse_db(pred_c[i],ref_c[i])
                     for i in range(len(pred_c))])

nmse_in = np.mean([compute_nmse_db(in_c[i],ref_c[i])
                   for i in range(len(in_c))])

print("Baseline NMSE (HPA output vs target): {:.2f} dB".format(nmse_in))
print("DPD NMSE: {:.2f} dB".format(nmse_pred))