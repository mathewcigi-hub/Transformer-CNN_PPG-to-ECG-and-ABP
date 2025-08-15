import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout, Input, Reshape, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
import numpy as np


def positional_encoding(position, d_model):
    angle_rads = np.arange(position)[:, np.newaxis] * np.arange(d_model)[np.newaxis, :] \
                 / np.power(10000, (2 * (np.arange(d_model) // 2)) / np.float32(d_model))
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.constant(angle_rads[np.newaxis, ...], dtype=tf.float32)

# Transformer Encoder
def transformer_encoder(input_tensor, d_model, num_heads, ff_dim, dropout_rate=0.1):
    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(input_tensor, input_tensor)
    attn_output = Dropout(dropout_rate)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(input_tensor + attn_output)
    ffn = Dense(ff_dim, activation='relu')(out1)
    ffn = Dense(d_model)(ffn)
    ffn_output = Dropout(dropout_rate)(ffn)
    return LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

# CNN Decoder
def cnn_decoder(input_tensor):
    conv1 = Conv2D(64, (5, 5), activation='relu', padding='same')(input_tensor)
    pool1 = MaxPooling2D(pool_size=(1, 2))(conv1)
    conv2 = Conv2D(128, (1, 5), activation='relu', padding='valid')(pool1) 
    conv3 = Conv2D(1, (5, 5), activation='relu', padding='same')(conv2)
    conv3_reshape = tf.reshape(conv3, (-1, 1, 1250, 1))
    return conv3_reshape

# Proposed Transformer + Dual CNN Model
def Proposed_model():
    input_length = 1250   # 10s @ 125Hz
    embedding_size = 125  # 1s per embedding
    num_embeddings = 10
    num_encoders = 8
    num_heads = 8
    d_model = 125
    ff_dim = 512


    input_signal = Input(shape=(input_length, 1), name="PPG_Input")

    x = Reshape((num_embeddings, embedding_size))(input_signal)
    x += positional_encoding(num_embeddings, embedding_size)

    encoder_outputs = []
    current_output = x
    for i in range(num_encoders):
        current_output = transformer_encoder(current_output, d_model, num_heads, ff_dim)
        encoder_outputs.append(current_output)


    concat_output = tf.concat(encoder_outputs, axis=-1)  # shape: (batch, 10, 125*8)
    concat_output = tf.reshape(concat_output, (-1, input_length, num_encoders))  # (batch, 1250, 8)
    concat_output = tf.expand_dims(concat_output, -1)  # (batch, 1250, 8, 1)


    wa = [tf.Variable(1.0, trainable=True, dtype=tf.float32, name=f"wa{i+1}") for i in range(num_encoders)]
    wa_tensor = tf.stack(wa)  # shape: (8,)
    abp_weighted = concat_output * wa_tensor  


    wb = [tf.Variable(1.0, trainable=True, dtype=tf.float32, name=f"wb{i+1}") for i in range(num_encoders)]
    wb_tensor = tf.stack(wb)
    ecg_weighted = concat_output * wb_tensor

    abp_output = cnn_decoder(abp_weighted)
    ecg_output = cnn_decoder(ecg_weighted)


    model = Model(inputs=input_signal, outputs=[abp_output, ecg_output], name="DualOutput_Transformer_CNN")
    return model
