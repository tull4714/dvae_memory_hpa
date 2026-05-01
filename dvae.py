import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

# Sampling layer
@tf.keras.utils.register_keras_serializable()
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

@tf.keras.utils.register_keras_serializable()
class DVAE(models.Model):
    def __init__(self, encoder, decoder, beta=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.sampling = Sampling()

    # Add an explicit build method for custom subclassed models
    def build(self, input_shape):
        # Call the build methods of sub-models if they are not already built
        if not self.encoder.built:
            self.encoder.build(input_shape)
        # The decoder's input shape is (batch_size, latent_dim)
        # We need to get latent_dim from the encoder's output or pass it directly.
        # Assuming latent_dim is accessible or set during DVAE init for decoder construction.
        # A simpler way to ensure the decoder is built is to define its input_shape based on latent_dim.
        # Since build_decoder takes latent_dim, we can ensure it's built there.
        # For the DVAE itself, simply call super().build.
        super().build(input_shape)
        
    # Add get_config to serialize the model
    def get_config(self):
        config = super().get_config()
        # Save the configurations of the sub-models
        config.update({
            "encoder_config": self.encoder.get_config(),
            "decoder_config": self.decoder.get_config(),
            "beta": self.beta,
        })
        return config

    # Add from_config to deserialize the model
    @classmethod
    def from_config(cls, config, custom_objects=None):
        beta = config.pop("beta", 1e-3)
        encoder_config_dict = config.pop("encoder_config")
        decoder_config_dict = config.pop("decoder_config")

        # Make sure custom_objects are passed to sub-model deserialization
        if custom_objects is None:
            custom_objects = {}
        # Ensure Sampling layer is recognized during sub-model deserialization
        custom_objects['Sampling'] = Sampling

        # Reconstruct encoder and decoder using Model.from_config
        reconstructed_encoder = tf.keras.models.Model.from_config(encoder_config_dict, custom_objects=custom_objects)
        reconstructed_decoder = tf.keras.models.Model.from_config(decoder_config_dict, custom_objects=custom_objects)

        # Instantiate the DVAE class with the reconstructed sub-models and beta
        return cls(encoder=reconstructed_encoder, decoder=reconstructed_decoder, beta=beta, **config)

    def compile(self, optimizer, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer
        self.loss_tracker = {
            "loss": tf.keras.metrics.Mean(name="loss"),
            "recon_loss": tf.keras.metrics.Mean(name="recon_loss"),
            "kl_loss": tf.keras.metrics.Mean(name="kl_loss")
        }

    @property
    def metrics(self):
        return list(self.loss_tracker.values())

    def train_step(self, data):
        # data: (x_out, x_in)
        x_out, x_in = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(x_out, training=True)
            z = self.sampling([z_mean, z_log_var])
            x_pred = self.decoder(z, training=True)
            # reconstruction: MSE over real+imag channels
            recon_loss = tf.reduce_mean(tf.square(x_in - x_pred))
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            loss = recon_loss + self.beta * kl_loss
        grads = tape.gradient(loss, self.encoder.trainable_weights + self.decoder.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.encoder.trainable_weights + self.decoder.trainable_weights))
        # update metrics
        self.loss_tracker["loss"].update_state(loss)
        self.loss_tracker["recon_loss"].update_state(recon_loss)
        self.loss_tracker["kl_loss"].update_state(kl_loss)
        return {k: v.result() for k, v in self.loss_tracker.items()}

    def test_step(self, data):
        x_out, x_in = data
        z_mean, z_log_var = self.encoder(x_out, training=False)
        z = self.sampling([z_mean, z_log_var])
        x_pred = self.decoder(z, training=False)
        recon_loss = tf.reduce_mean(tf.square(x_in - x_pred))
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        loss = recon_loss + self.beta * kl_loss
        self.loss_tracker["loss"].update_state(loss)
        self.loss_tracker["recon_loss"].update_state(recon_loss)
        self.loss_tracker["kl_loss"].update_state(kl_loss)
        return {k: v.result() for k, v in self.loss_tracker.items()}

    def call(self, inputs, training=False):
        z_mean, z_log_var = self.encoder(inputs, training=training)

        if training:
            # 학습 중에는 VAE 방식 유지
            z = self.sampling([z_mean, z_log_var])
        else:
            # 추론 시에는 deterministic하게 평균값 사용
            z = z_mean

        return self.decoder(z, training=training)