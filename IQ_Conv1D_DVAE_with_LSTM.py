# IQ_Conv1D_DPD_Deterministic.py
# Improved Deterministic DPD (SOTA-style)

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import re # Added for regular expressions

# --------------------------------------------------
# 1. Utilities
# --------------------------------------------------

def normalize_pair(I, Q, ref_rms=None):
    mag = np.sqrt(I**2 + Q**2)
    if ref_rms is None:
        ref_rms = np.sqrt(np.mean(mag**2) + 1e-12)
    return I/ref_rms, Q/ref_rms, ref_rms

def to_2ch(i, q):
    return np.stack([i, q], axis=-1).astype(np.float32)

def to_complex(x2):
    return x2[...,0] + 1j*x2[...,1]

# --------------------------------------------------
# 2. Complex NMSE loss (DPD-friendly)
# --------------------------------------------------

def complex_nmse_loss(y_true, y_pred):
    err = y_true - y_pred
    power = tf.reduce_mean(tf.square(y_true))
    return tf.reduce_mean(tf.square(err)) / (power + 1e-9)

# --------------------------------------------------
# 3. Deterministic DPD Model (SOTA-style)
# --------------------------------------------------

def build_dpd_model(seq_len):

    inp = layers.Input(shape=(seq_len, 2))

    x = layers.Conv1D(256, 7, padding="same", activation="relu")(inp)
    x = layers.Conv1D(256, 7, padding="same", activation="relu")(x)

    x = layers.LSTM(256, return_sequences=True)(x)

    x = layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)

    correction = layers.Conv1D(2, 1, padding="same")(x)

    # residual scaling (IMPORTANT)
    correction = layers.Lambda(lambda t: 0.1 * t)(correction)

    out = layers.Add()([inp, correction])

    return models.Model(inp, out, name="SOTA_DPD")

# --------------------------------------------------
# 4. Data Loading
# --------------------------------------------------

seq_len = 64
batch_size = 64

file1_I = '/content/drive/MyDrive/input_iq_I.csv'
file1_Q = '/content/drive/MyDrive/input_iq_Q.csv'
file2_I = '/content/drive/MyDrive/input_target_I.csv'
file2_Q = '/content/drive/MyDrive/input_target_Q.csv'

# load csv
in_I = pd.read_csv(file1_I,header=None).to_numpy()
in_Q = pd.read_csv(file1_Q,header=None).to_numpy()
t_I  =  pd.read_csv(file2_I,header=None).to_numpy()
t_Q  =  pd.read_csv(file2_Q,header=None).to_numpy()

# Reshape data to (num_sequences, seq_len)
# Assuming original numpy arrays are (total_samples, 1)
num_total_samples_I = in_I.shape[0]
if num_total_samples_I % seq_len != 0:
    raise ValueError(f"Total samples for I ({num_total_samples_I}) must be divisible by seq_len ({seq_len}) for reshaping.")

num_total_samples_Q = in_Q.shape[0]
if num_total_samples_Q % seq_len != 0:
    raise ValueError(f"Total samples for Q ({num_total_samples_Q}) must be divisible by seq_len ({seq_len})
    
# reshape
in_I = in_I.reshape(-1, seq_len)
in_Q = in_Q.reshape(-1, seq_len)
t_I  = t_I.reshape(-1, seq_len)
t_Q  = t_Q.reshape(-1, seq_len)

# split
N = in_I.shape[0]
n_train = int(0.8*N)
n_val   = int(0.1*N)

I_tr, Q_tr = in_I[:n_train], in_Q[:n_train]
I_va, Q_va = in_I[n_train:n_train+n_val], in_Q[n_train:n_train+n_val]
I_te, Q_te = in_I[n_train+n_val:], in_Q[n_train+n_val:]

tI_tr, tQ_tr = t_I[:n_train], t_Q[:n_train]
tI_va, tQ_va = t_I[n_train:n_train+n_val], t_Q[n_train:n_train+n_val]
tI_te, tQ_te = t_I[n_train+n_val:], t_Q[n_train+n_val:]

# RMS normalization (CRITICAL FIX)
I_tr, Q_tr, rms = normalize_pair(I_tr, Q_tr)
tI_tr, tQ_tr, _ = normalize_pair(tI_tr, tQ_tr, rms)

I_va, Q_va, _ = normalize_pair(I_va, Q_va, rms)
tI_va, tQ_va, _ = normalize_pair(tI_va, tQ_va, rms)

I_te, Q_te, _ = normalize_pair(I_te, Q_te, rms)
tI_te, tQ_te, _ = normalize_pair(tI_te, tQ_te, rms)

X_tr = to_2ch(I_tr, Q_tr)
Y_tr = to_2ch(tI_tr, tQ_tr)

X_va = to_2ch(I_va, Q_va)
Y_va = to_2ch(tI_va, tQ_va)

X_te = to_2ch(I_te, Q_te)
Y_te = to_2ch(tI_te, tQ_te)

# --------------------------------------------------
# 5. Train
# --------------------------------------------------

model = build_dpd_model(seq_len)
model.compile(
    optimizer=tf.keras.optimizers.Adam(2e-4),
    loss=complex_nmse_loss
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
  
    model.fit(
        X_tr, Y_tr,
        validation_data=(X_va, Y_va),
        epochs=current_epochs_to_run,
        batch_size=batch_size
    )
    
    # --------------------------------------------------
    # 6. Save model
    # --------------------------------------------------
    model_save_path_keras = os.path.join(model_dir, f'my_dpd_model_{target_save_epoch_total}.keras') # Changed filename
    model.save(model_save_path_keras) # 확장자를 .keras로 변경
    print(f'DPD 모델이 {model_save_path_keras}에 저장되었습니다.') # Updated print
    print("evaluate shape: ", X_te.shape, Y_te.shape)
    model.evaluate(X_te, Y_te)

# -------------------------
# 7. Evaluation
# -------------------------

pred = model.predict(X_te)

pred_c = to_complex(pred)
ref_c = to_complex(Y_te)
in_c = to_complex(X_te)

nmse_pred = np.mean([complex_nmse_loss(pred_c[i],ref_c[i])
                     for i in range(len(pred_c))])

nmse_in = np.mean([complex_nmse_loss(in_c[i],ref_c[i])
                   for i in range(len(in_c))])

print("Baseline NMSE (HPA output vs target): {:.2f} dB".format(nmse_in))
print("DPD NMSE: {:.2f} dB".format(nmse_pred))