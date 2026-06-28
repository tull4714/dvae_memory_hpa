import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# ==========================================
# 1. 하이퍼파라미터 및 설정
# ==========================================
seq_len = 64          # OFDM 심볼의 서브블록 길이 (데이터 생성 코드의 N과 동일해야 함)
d_model = 64          # 트랜스포머 내부 차원 (IQ 2채널을 64차원으로 확장하여 분석)
num_heads = 4         # 멀티 헤드 어텐션 개수
ff_dim = 128          # 피드포워드 네트워크 은닉층 크기
num_layers = 2        # 트랜스포머 블록(인코더) 개수
batch_size = 512
epochs = 100
learning_rate = 1e-3

data_dir = '/content/drive/MyDrive/NonlinearMemory/'
model_dir = '/content/drive/MyDrive/NonlinearMemory/'

# ==========================================
# 2. 트랜스포머 구성 요소 정의
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
    # 1. Multi-Head Attention
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inputs, inputs)
    attn_output = layers.Dropout(dropout)(attn_output)
    out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    # 2. Feed Forward Network
    ffn_output = layers.Dense(ff_dim, activation="relu")(out1)
    ffn_output = layers.Dense(d_model)(ffn_output)
    ffn_output = layers.Dropout(dropout)(ffn_output)
    return layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

def build_transformer_dpd(seq_len, d_model, num_heads, ff_dim, num_layers):
    """전체 트랜스포머 DPD 모델 조립"""
    inputs = layers.Input(shape=(seq_len, 2))
    
    # 2채널(IQ)을 d_model 차원으로 투영
    x = layers.Dense(d_model)(inputs)
    x = PositionalEncoding(seq_len, d_model)(x)
    
    # 트랜스포머 블록 반복
    for _ in range(num_layers):
        x = transformer_encoder(x, d_model, num_heads, ff_dim)
    
    # 다시 2채널(IQ)로 축소하여 최종 보정된 신호 출력
    outputs = layers.Dense(2, activation="linear")(x)
    
    return models.Model(inputs=inputs, outputs=outputs, name="Transformer_DPD")

# ==========================================
# 3. 데이터 로드 및 전처리 (기존과 동일)
# ==========================================
print("데이터 로딩 중...")
df_I = pd.read_csv(os.path.join(data_dir, 'inputs_I.csv'), header=None)
df_Q = pd.read_csv(os.path.join(data_dir, 'inputs_Q.csv'), header=None)
df_tI = pd.read_csv(os.path.join(data_dir, 'targets_I.csv'), header=None)
df_tQ = pd.read_csv(os.path.join(data_dir, 'targets_Q.csv'), header=None)

# 형태 변환: (전체 샘플수, 2)
input_data = np.stack((df_I.values.flatten(), df_Q.values.flatten()), axis=-1)
target_data = np.stack((df_tI.values.flatten(), df_tQ.values.flatten()), axis=-1)

# RMS 정규화 (Target/Clean 신호 기준!)
rms_train = np.sqrt(np.mean(target_data**2))
print(f"✅ 정규화 기준 (Target RMS): {rms_train}")
input_data_norm = input_data / rms_train
target_data_norm = target_data / rms_train

# 시퀀스 단위(seq_len)로 분할
num_samples = (len(input_data_norm) // seq_len) * seq_len
X_train = input_data_norm[:num_samples].reshape(-1, seq_len, 2)
y_train = target_data_norm[:num_samples].reshape(-1, seq_len, 2)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

# ==========================================
# 4. 모델 빌드 및 가중치 복구 로직
# ==========================================
transformer = build_transformer_dpd(seq_len, d_model, num_heads, ff_dim, num_layers)
transformer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='mse')

# 저장된 모델 찾기 (.weights.h5 확장자 사용 권장)
saved_models = [f for f in os.listdir(model_dir) if f.startswith('transformer_dpd_') and f.endswith('.weights.h5')]
latest_epoch = 0

if saved_models:
    epoch_numbers = [int(re.search(r'_(\d+)\.weights\.h5', f).group(1)) for f in saved_models if re.search(r'_(\d+)\.weights\.h5', f)]
    if epoch_numbers:
        latest_epoch = max(epoch_numbers)
        latest_model_path = os.path.join(model_dir, f'transformer_dpd_{latest_epoch}.weights.h5')
        print(f"✅ 저장된 트랜스포머 모델 발견: {latest_model_path}. (에포크 {latest_epoch}에서 재개)")
        transformer.load_weights(latest_model_path)
else:
    print("새로운 트랜스포머 모델 훈련을 시작합니다.")

transformer.summary()

# ==========================================
# 5. 모델 훈련 및 저장
# ==========================================
class SaveModelCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        current_total_epoch = latest_epoch + epoch + 1
        if current_total_epoch % 10 == 0:
            save_path = os.path.join(model_dir, f'transformer_dpd_{current_total_epoch}.weights.h5')
            self.model.save_weights(save_path)
            print(f"\n[Epoch {current_total_epoch}] 가중치 저장 완료: {save_path}")

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

print("훈련 시작...")
history = transformer.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    callbacks=[SaveModelCallback()]
)