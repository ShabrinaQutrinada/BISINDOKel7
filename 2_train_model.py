import json
import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

import config
from utils import NUM_FEATURES

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_dataset():
    X, y, labels = [], [], []
    for word in config.ACTIONS:
        folder = os.path.join(config.DATA_PATH, word)
        if not os.path.isdir(folder):
            print(f"  [peringatan] folder {folder} tidak ada, kata '{word}' dilewati")
            continue
        files = sorted(f for f in os.listdir(folder) if f.endswith(".npy"))
        if not files:
            print(f"  [peringatan] {folder} kosong, kata '{word}' dilewati")
            continue
        label_idx = len(labels)
        labels.append(word)
        for f in files:
            seq = np.load(os.path.join(folder, f))
            if seq.shape != (config.SEQUENCE_LENGTH, NUM_FEATURES):
                print(f"  [peringatan] ukuran {f} tidak sesuai {seq.shape}, dilewati")
                continue
            X.append(seq)
            y.append(label_idx)
        print(f"  {word:16s}: {len(files)} rekaman")
    return np.array(X, dtype=np.float32), np.array(y), labels


def augment(X, y, copies=2):
    """Menambah variasi data latih: noise kecil, skala, dan pergeseran posisi."""
    X_aug, y_aug = [X], [y]
    for _ in range(copies):
        noise = np.random.normal(0, 0.02, X.shape).astype(np.float32)
        scale = np.random.uniform(0.9, 1.1, (len(X), 1, 1)).astype(np.float32)
        shift = np.random.uniform(-0.1, 0.1, (len(X), 1, 1)).astype(np.float32)
        Xn = X * scale + shift + noise
        Xn[X == 0] = 0  # tangan yang tidak terdeteksi tetap nol
        X_aug.append(Xn)
        y_aug.append(y)
    return np.concatenate(X_aug), np.concatenate(y_aug)


def build_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(config.SEQUENCE_LENGTH, NUM_FEATURES)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(128),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def save_history_plot(history, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history.history["accuracy"], label="train")
    ax[0].plot(history.history["val_accuracy"], label="validasi")
    ax[0].set_title("Akurasi")
    ax[0].set_xlabel("epoch")
    ax[0].legend()
    ax[1].plot(history.history["loss"], label="train")
    ax[1].plot(history.history["val_loss"], label="validasi")
    ax[1].set_title("Loss")
    ax[1].set_xlabel("epoch")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Grafik training disimpan: {path}")


def main():
    print("Memuat dataset...")
    X, y, labels = load_dataset()
    if len(labels) < 2:
        raise SystemExit("Minimal perlu 2 kata yang sudah direkam. Jalankan 1_collect_data.py dulu.")
    print(f"Total: {len(X)} rekaman, {len(labels)} kelas, bentuk data {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    X_train, y_train = augment(X_train, y_train)
    print(f"Data latih setelah augmentasi: {len(X_train)}, data uji: {len(X_test)}")

    model = build_model(len(labels))
    model.summary()

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    cb = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=30,
                                restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10),
    ]
    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=200, batch_size=16, callbacks=cb, verbose=2)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nAkurasi data uji: {acc * 100:.2f}%")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=labels, zero_division=0))
    print("Confusion matrix (baris = label asli, kolom = prediksi):")
    print("labels:", labels)
    print(confusion_matrix(y_test, y_pred))

    model.save(config.MODEL_PATH)
    with open(config.LABELS_PATH, "w") as f:
        json.dump(labels, f)
    print(f"\nModel disimpan: {config.MODEL_PATH}")
    print(f"Label disimpan: {config.LABELS_PATH}")

    save_history_plot(history, os.path.join(config.MODEL_DIR, "training_history.png"))


if __name__ == "__main__":
    main()
