import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


# -----------------------------------------------------------------------
# 전체 메모리 다항식 계수 (q=0,1,2 / k=0,2,4)
# -----------------------------------------------------------------------
ALL_C_VALS = {
    (0, 0):  1.0       + 0j,
    (2, 0): -0.00542   - 0.02900j,
    (4, 0): -0.009657  - 0.007028j,
    (0, 1): -0.00680   - 0.00023j,
    (2, 1):  0.02234   + 0.02317j,
    (4, 1): -0.002451  - 0.003735j,
    (0, 2):  0.00289   - 0.00054j,
    (2, 2): -0.00621   - 0.00932j,
    (4, 2):  0.001229  + 0.001508j,
}


# -----------------------------------------------------------------------
# TF 미분 가능 HPA
# train_step 내부에서 호출되어 predistorted → HPA 경로의 gradient를
# 모델까지 전달한다. (실수 연산만 사용 → gradient 안정)
# -----------------------------------------------------------------------
def polynomial_tf(p_in_2ch, backoff=5.0, snr_db=-1.0, memory_depth=2):
    c_vals = {(k, q): v for (k, q), v in ALL_C_VALS.items()
              if q <= memory_depth}

    backoff_linear = tf.constant(10.0 ** (backoff / 10.0), dtype=tf.float32)

    p_real   = p_in_2ch[..., 0]
    p_imag   = p_in_2ch[..., 1]
    p_mag_sq = p_real ** 2 + p_imag ** 2
    p_angle  = tf.math.atan2(p_imag, p_real)

    data_tx_avg = tf.reduce_mean(p_mag_sq, axis=1, keepdims=True)
    denom       = data_tx_avg * backoff_linear + 1e-10
    tx_amp      = tf.sqrt(p_mag_sq / denom)

    p_out_real = tf.zeros_like(tx_amp)
    p_out_imag = tf.zeros_like(tx_amp)

    for (k_idx, q_idx), c_val in c_vals.items():
        c_r = float(np.real(c_val))
        c_i = float(np.imag(c_val))

        if q_idx == 0:
            tx_d = tx_amp
        else:
            pad  = tf.zeros([tf.shape(tx_amp)[0], q_idx], dtype=tf.float32)
            tx_d = tf.concat([pad, tx_amp[:, :-q_idx]], axis=1)

        term = tx_d if k_idx == 0 else tx_d * tf.pow(tx_d + 1e-10, float(k_idx))

        p_out_real = p_out_real + c_r * term
        p_out_imag = p_out_imag + c_i * term

    scale    = tf.sqrt(data_tx_avg * backoff_linear)
    cos_a    = tf.cos(p_angle)
    sin_a    = tf.sin(p_angle)
    hpa_real = (p_out_real * cos_a - p_out_imag * sin_a) * scale
    hpa_imag = (p_out_real * sin_a + p_out_imag * cos_a) * scale

    if snr_db >= 0.0:
        signal_power = tf.reduce_mean(hpa_real ** 2 + hpa_imag ** 2)
        snr_linear   = tf.constant(10.0 ** (snr_db / 10.0), dtype=tf.float32)
        noise_std    = tf.sqrt(signal_power / snr_linear / 2.0)
        hpa_real    += tf.random.normal(tf.shape(hpa_real), stddev=noise_std)
        hpa_imag    += tf.random.normal(tf.shape(hpa_imag), stddev=noise_std)

    return tf.stack([hpa_real, hpa_imag], axis=-1)


# -----------------------------------------------------------------------
# PhysicsBasisLayer — 물리 기저(메모리 다항식) 생성 레이어
#
# 입력 : (batch, seq_len, 2)  — [I, Q]
# 출력 : (batch, seq_len, 2 + 2*K*Q)  — 원본 + 비선형 메모리 기저
#
# 메모리 다항식 항 x[n-q]·|x[n-q]|^k 를 미리 계산해서 채널로 붙임.
# 신경망은 이 기저들의 조합 계수만 학습하면 되므로 수렴이 빠르다.
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
        mag = tf.sqrt(I ** 2 + Q ** 2 + 1e-12)

        feats = [I, Q]

        for q in range(self.q_depth):
            if q == 0:
                I_d, Q_d, mag_d = I, Q, mag
            else:
                pad = tf.zeros([tf.shape(I)[0], q], dtype=I.dtype)
                I_d   = tf.concat([pad, I[:, :-q]], axis=1)
                Q_d   = tf.concat([pad, Q[:, :-q]], axis=1)
                mag_d = tf.concat([pad, mag[:, :-q]], axis=1)

            for k in self.k_orders:
                gain = tf.pow(mag_d + 1e-12, float(k))
                feats.append(I_d * gain)
                feats.append(Q_d * gain)

        return tf.stack(feats, axis=-1)

    def compute_output_shape(self, input_shape):
        num_feats = 2 + 2 * len(self.k_orders) * self.q_depth
        return (input_shape[0], input_shape[1], num_feats)

    def get_config(self):
        config = super().get_config()
        config.update({"k_orders": list(self.k_orders), "q_depth": self.q_depth})
        return config


# -----------------------------------------------------------------------
# CurriculumScheduler
#
# 이제 train_step에서 실제로 polynomial_tf(memory_depth)를 통과하므로
# memory_depth가 학습에 직접 영향을 준다 (커리큘럼이 의미를 가짐).
#   Stage 0: memory_depth=0 → 메모리 없는 비선형만 보상 학습
#   Stage 1: memory_depth=1 → 1-tap 메모리 추가
#   Stage 2: memory_depth=2 → 전체 메모리 (실제 HPA)
# -----------------------------------------------------------------------
class CurriculumScheduler:
    STAGES = [
        {"memory_depth": 0, "snr_db": -1.0,
         "description": "Stage 0 | memory=0tap (30 epochs)"},
        {"memory_depth": 1, "snr_db": -1.0,
         "description": "Stage 1 | memory=1tap (30 epochs)"},
        {"memory_depth": 2, "snr_db": -1.0,
         "description": "Stage 2 | memory=2tap (40 epochs)"},
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
            print(f"\n{'='*50}")
            print(f"커리큘럼 단계 전환 → {self.description}")
            print(f"{'='*50}\n")
        else:
            print("이미 최종 커리큘럼 단계입니다.")

    def status(self):
        print(f"현재 커리큘럼: {self.description}")


# -----------------------------------------------------------------------
# DVAE_Curriculum (DLA — 실제 HPA 통과 방식, 결정론적 AE)
#
# ★ 핵심 변경:
#   기존(Supervised): loss = MSE(correction_target - correction)
#                     → correction_target = 원본 - HPA(원본) (1차 근사, 부정확)
#   변경(DLA):        loss = MSE(원본 - HPA(predistorted))
#                     → 추론 EVM과 정확히 동일한 목표
#
# 구조:
#   원본 → PhysicsBasis+TCN(encoder) → TCN(decoder) → correction
#   predistorted = 원본 + correction
#   loss = MSE(원본 - HPA(predistorted))
#
# 훈련 데이터:
#   X_in = 원본,  y(X_out)는 사용하지 않음 (HPA를 train_step에서 직접 통과)
# -----------------------------------------------------------------------
@tf.keras.utils.register_keras_serializable()
class DVAE_Curriculum(models.Model):

    def __init__(self, encoder, decoder,
                 backoff=5.0,
                 memory_depth=0, snr_db=-1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.encoder       = encoder
        self.decoder       = decoder
        self.backoff       = backoff
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
            "memory_depth":   self._memory_depth,
            "snr_db":         self._snr_db,
        })
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        backoff      = config.pop("backoff",      5.0)
        memory_depth = config.pop("memory_depth", 0)
        snr_db       = config.pop("snr_db",       -1.0)
        enc_cfg      = config.pop("encoder_config")
        dec_cfg      = config.pop("decoder_config")

        if custom_objects is None:
            custom_objects = {}
        custom_objects['PhysicsBasisLayer'] = PhysicsBasisLayer

        encoder = tf.keras.models.Model.from_config(
            enc_cfg, custom_objects=custom_objects)
        decoder = tf.keras.models.Model.from_config(
            dec_cfg, custom_objects=custom_objects)

        return cls(encoder=encoder, decoder=decoder,
                   backoff=backoff,
                   memory_depth=memory_depth, snr_db=snr_db, **config)

    def compile(self, optimizer, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer
        self.loss_tracker = {
            "loss": tf.keras.metrics.Mean(name="loss"),
        }

    @property
    def metrics(self):
        return list(self.loss_tracker.values())

    def call(self, inputs, training=False):
        """
        원본 신호 → predistorted 신호
        추론 시 이 출력을 HPA에 통과시키면 원본에 가까운 신호가 나온다.
        """
        z            = self.encoder(inputs, training=training)
        correction   = self.decoder(z, training=training)
        predistorted = inputs + correction
        return predistorted

    def train_step(self, data):
        """
        DLA train_step (실제 HPA 통과):
          x = 원본 신호
          predistorted = 원본 + correction
          loss = MSE(원본 - HPA(predistorted))

        gradient가 HPA → decoder → encoder 전체 경로로 흐른다.
        커리큘럼 memory_depth에 따라 HPA 메모리 차수가 단계적으로 증가.
        (y는 데이터 로더 호환을 위해 받기만 하고 사용하지 않음)
        """
        x, y = data

        with tf.GradientTape() as tape:
            predistorted = self(x, training=True)

            # 실제 HPA 통과 — 현재 커리큘럼 단계의 memory_depth 사용
            hpa_output = polynomial_tf(
                predistorted,
                backoff      = self.backoff,
                snr_db       = self._snr_db,
                memory_depth = self._memory_depth
            )

            # HPA 출력이 원본에 가까워지도록 (= 추론 EVM 목표와 동일)
            loss = tf.reduce_mean(tf.square(x - hpa_output))

        grads = tape.gradient(
            loss,
            self.encoder.trainable_weights + self.decoder.trainable_weights
        )
        self.optimizer.apply_gradients(zip(
            grads,
            self.encoder.trainable_weights + self.decoder.trainable_weights
        ))

        self.loss_tracker["loss"].update_state(loss)
        return {k: v.result() for k, v in self.loss_tracker.items()}

    def test_step(self, data):
        x, y = data

        predistorted = self(x, training=False)
        hpa_output = polynomial_tf(
            predistorted,
            backoff      = self.backoff,
            snr_db       = self._snr_db,
            memory_depth = self._memory_depth
        )
        loss = tf.reduce_mean(tf.square(x - hpa_output))

        self.loss_tracker["loss"].update_state(loss)
        return {k: v.result() for k, v in self.loss_tracker.items()}
