import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from Proposed_model import Transformer_model  


# PATHS
DATASET_PATH = " <data location>"
FINAL_MODEL_PATH = "<To where the final model location>"
BEST_MODEL_PATH = "<To save best model location>"
EPOCH_FILE = "<To epoch number file>"


def rpeak_weighted_mse(y_true, y_pred, rpeak_mask):
    weights = 1.0 + 9.0 * rpeak_mask
    return tf.reduce_mean(weights * tf.square(y_true - y_pred))


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_list, dataset_path, batch_size=16, shuffle=True):
        self.file_list = file_list
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.file_list) / self.batch_size))

    def __getitem__(self, index):
        batch_files = self.file_list[index * self.batch_size:(index + 1) * self.batch_size]
        ppg_batch, rpeak_batch, abp_batch, ecg_batch = [], [], [], []

        for file_name in batch_files:
            with open(os.path.join(self.dataset_path, file_name), 'r') as file:
                data = json.load(file)
                ppg_data = np.array(data['PPG_derivatives_values'])[:, 0]

                if len(ppg_data) == 1250:
                    # R-peak mask
                    rpeak_mask = np.zeros(1250, dtype=np.float32)
                    for r_index in data['ECG_Rpeak']:
                        if 0 <= int(r_index) < 1250:
                            rpeak_mask[int(r_index)] = 1.0

                    if len(data['abp_raw_values']) == 1250 and len(data['ECG_f_values']) == 1250:
                        ppg_batch.append(ppg_data)
                        rpeak_batch.append(rpeak_mask)
                        abp_batch.append(data['abp_raw_values'])
                        ecg_batch.append(data['ECG_f_values'])

        ppg_batch = np.array(ppg_batch, dtype=np.float32)[..., np.newaxis]
        rpeak_batch = np.array(rpeak_batch, dtype=np.float32)[..., np.newaxis]
        abp_batch = np.array(abp_batch, dtype=np.float32)[..., np.newaxis]
        ecg_batch = np.array(ecg_batch, dtype=np.float32)[..., np.newaxis]

        return (ppg_batch, rpeak_batch), (abp_batch, ecg_batch)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.file_list)


def save_epoch(epoch):
    with open(EPOCH_FILE, 'w') as f:
        f.write(str(epoch))

def load_epoch():
    if os.path.exists(EPOCH_FILE):
        with open(EPOCH_FILE, 'r') as f:
            return int(f.read())
    return 0

class EpochCheckpoint(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        save_epoch(epoch + 1)


class CombinedModel(tf.keras.Model):
    def train_step(self, data):
        (ppg, rpeak_mask), (y_abp, y_ecg) = data
        with tf.GradientTape() as tape:
            pred_abp, pred_ecg = self([ppg, rpeak_mask], training=True)
            loss_abp = tf.reduce_mean(tf.abs(y_abp - pred_abp))  # MAE
            loss_ecg = rpeak_weighted_mse(y_ecg, pred_ecg, rpeak_mask)
            total_loss = loss_abp + loss_ecg

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": total_loss, "loss_abp": loss_abp, "loss_ecg": loss_ecg}

    def test_step(self, data):
        (ppg, rpeak_mask), (y_abp, y_ecg) = data
        pred_abp, pred_ecg = self([ppg, rpeak_mask], training=False)
        loss_abp = tf.reduce_mean(tf.abs(y_abp - pred_abp))
        loss_ecg = rpeak_weighted_mse(y_ecg, pred_ecg, rpeak_mask)
        total_loss = loss_abp + loss_ecg
        return {"loss": total_loss, "loss_abp": loss_abp, "loss_ecg": loss_ecg}

if __name__ == "__main__":
    print("Loading dataset...")
    file_list = [f for f in os.listdir(DATASET_PATH) if f.endswith('.json')]
    train_files, val_files = train_test_split(file_list, test_size=0.2, random_state=42)

    train_gen = DataGenerator(train_files, DATASET_PATH, batch_size=16)
    val_gen = DataGenerator(val_files, DATASET_PATH, batch_size=16, shuffle=False)

    print("Building model from Transformer_model...")
    ppg_input = tf.keras.layers.Input(shape=(1250, 1), name="PPG_Input")
    rpeak_input = tf.keras.layers.Input(shape=(1250, 1), name="RPeak_Input")

    abp_out, ecg_out = Transformer_model(ppg_input, rpeak_input)  # must return 2 outputs
    model = CombinedModel(inputs=[ppg_input, rpeak_input], outputs=[abp_out, ecg_out])

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))

    os.makedirs(os.path.dirname(FINAL_MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)

    initial_epoch = load_epoch()

    callbacks = [
        ModelCheckpoint(BEST_MODEL_PATH, save_best_only=True, monitor="val_loss", mode="min"),
        EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=10, verbose=1),
        EpochCheckpoint()
    ]

    print("Training model...")
    history = model.fit(train_gen, validation_data=val_gen, epochs=25,
                        initial_epoch=initial_epoch, callbacks=callbacks)

    print("Saving final model...")
    model.save(FINAL_MODEL_PATH, overwrite=True)
