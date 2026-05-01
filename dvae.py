import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models


# -----------------------------
# DLA: TensorFlow 미분 가능 HPA
# (batch, seq_len, 2) → (batch, seq_len, 2)
# train_step 내부의 GradientTape 안에서 호출되어
# predistorted → HPA 경로의 gradient를 DVAE까지 전달
# -----------------------------
def polynomial_tf(p_in_2ch, backoff=5.0):
    c_vals = {
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
    backoff_linear = tf.constant(10.0 ** (backoff / 10.0), dtype=tf.float32)

    p_real = p_in_2ch[..., 0]                                       # (batch, seq_len)
    p_imag = p_in_2ch[..., 1]

    # 시퀀스별 전력 정규화 (numpy polynomial()과 동일한 로직)
    p_mag_sq    = p_real ** 2 + p_imag ** 2                         # (batch, seq_len)
    data_tx_avg = tf.reduce_mean(p_mag_sq, axis=1, keepdims=True)   # (batch, 1)
    denom       = data_tx_avg * backoff_linear + 1e-10
    tx_amp      = tf.sqrt(p_mag_sq / denom)                         # (batch, seq_len)
    p_angle     = tf.math.atan2(p_imag, p_real)                     # (batch, seq_len)

    # 메모리 다항식 합산
    p_out_real = tf.zeros_like(tx_amp)
    p_out_imag = tf.zeros_like(tx_amp)

    for (k_idx, q_idx), c_val in c_vals.items():
        c_r = float(np.real(c_val))
        c_i = float(np.imag(c_val))

        # q_idx 샘플 delay (앞에 0 패딩, 뒤를 잘라냄)
        if q_idx == 0:
            tx_d = tx_amp
        else:
            pad  = tf.zeros([tf.shape(tx_amp)[0], q_idx], dtype=tf.float32)
            tx_d = tf.concat([pad, tx_amp[:, :-q_idx]], axis=1)

        # c[k,q] * tx_d * |tx_d|^k
        term = tx_d * tf.pow(tf.abs(tx_d) + 1e-10, float(k_idx))
        p_out_real = p_out_real + c_r * term
        p_out_imag = p_out_imag + c_i * term

    # 위상 복원 및 스케일 역변환
    scale    = tf.sqrt(data_tx_avg * backoff_linear)                # (batch, 1)
    cos_a    = tf.cos(p_angle)
    sin_a    = tf.sin(p_angle)
    hpa_real = (p_out_real * cos_a - p_out_imag * sin_a) * scale
    hpa_imag = (p_out_real * sin_a + p_out_imag * cos_a) * scale

    return tf.stack([hpa_real, hpa_imag], axis=-1)                  # (batch, seq_len, 2)


# -----------------------------
# Sampling Layer
# -----------------------------
@tf.keras.utils.register_keras_serializable()
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps


# -----------------------------
# DVAE (DLA 구조)
# -----------------------------
@tf.keras.utils.register_keras_serializable()
class DVAE(models.Model):

    def __init__(self, encoder, decoder, beta=1e-3, backoff=5.0, **kwargs):
        super().__init__(**kwargs)
        self.encoder  = encoder
        self.decoder  = decoder
        self.beta     = beta
        self.backoff  = backoff   # HPA backoff (dB) — polynomial_tf에 전달
        self.sampling = Sampling()

    def build(self, input_shape):
        if not self.encoder.built:
            self.encoder.build(input_shape)
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            "encoder_config": self.encoder.get_config(),
            "decoder_config": self.decoder.get_config(),
            "beta":    self.beta,
            "backoff": self.backoff,
        })
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        beta    = config.pop("beta",    1e-3)
        backoff = config.pop("backoff", 5.0)
        encoder_config_dict = config.pop("encoder_config")
        decoder_config_dict = config.pop("decoder_config")

        if custom_objects is None:
            custom_objects = {}
        custom_objects['Sampling'] = Sampling

        reconstructed_encoder = tf.keras.models.Model.from_config(
            encoder_config_dict, custom_objects=custom_objects)
        reconstructed_decoder = tf.keras.models.Model.from_config(
            decoder_config_dict, custom_objects=custom_objects)

        return cls(encoder=reconstructed_encoder, decoder=reconstructed_decoder,
                   beta=beta, backoff=backoff, **config)

    def compile(self, optimizer, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer
        self.loss_tracker = {
            "loss":       tf.keras.metrics.Mean(name="loss"),
            "recon_loss": tf.keras.metrics.Mean(name="recon_loss"),
            "kl_loss":    tf.keras.metrics.Mean(name="kl_loss"),
        }

    @property
    def metrics(self):
        return list(self.loss_tracker.values())

    def call(self, inputs, training=False):
        """
        DLA call:
          inputs  → encoder → z
          z       → decoder → correction
          outputs = inputs + correction   (predistorted 신호)
        추론 시 이 출력을 HPA에 통과시키면 원본에 가까운 신호가 나온다.
        """
        z_mean, z_log_var = self.encoder(inputs, training=training)

        if training:
            z = self.sampling([z_mean, z_log_var])
        else:
            z = z_mean                    # 추론: deterministic

        correction = self.decoder(z, training=training)
        predistorted = inputs + correction  # residual 구조
        return predistorted, z_mean, z_log_var

    def train_step(self, data):
        """
        DLA train_step:
          x = y = 원본 신호 (정규화됨)
          loss = MSE( HPA(predistorted), 원본 )
        gradient가 HPA → DVAE 전체 경로로 흐른다.
        """
        x, y = data   # x == y == 원본 신호

        with tf.GradientTape() as tape:
            predistorted, z_mean, z_log_var = self(x, training=True)

            # predistorted 신호를 TF HPA에 통과
            hpa_output = polynomial_tf(predistorted, self.backoff)

            # DLA reconstruction loss: HPA 출력 vs 원본
            recon_loss = tf.reduce_mean(tf.square(y - hpa_output))

            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            )
            loss = recon_loss + self.beta * kl_loss

        grads = tape.gradient(
            loss,
            self.encoder.trainable_weights + self.decoder.trainable_weights
        )
        self.optimizer.apply_gradients(zip(
            grads,
            self.encoder.trainable_weights + self.decoder.trainable_weights
        ))

        self.loss_tracker["loss"].update_state(loss)
        self.loss_tracker["recon_loss"].update_state(recon_loss)
        self.loss_tracker["kl_loss"].update_state(kl_loss)
        return {k: v.result() for k, v in self.loss_tracker.items()}

    def test_step(self, data):
        """
        DLA test_step: train_step과 동일한 loss 계산 (gradient 없음)
        """
        x, y = data

        predistorted, z_mean, z_log_var = self(x, training=False)
        hpa_output = polynomial_tf(predistorted, self.backoff)

        recon_loss = tf.reduce_mean(tf.square(y - hpa_output))
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        loss = recon_loss + self.beta * kl_loss

        self.loss_tracker["loss"].update_state(loss)
        self.loss_tracker["recon_loss"].update_state(recon_loss)
        self.loss_tracker["kl_loss"].update_state(kl_loss)
        return {k: v.result() for k, v in self.loss_tracker.items()}
