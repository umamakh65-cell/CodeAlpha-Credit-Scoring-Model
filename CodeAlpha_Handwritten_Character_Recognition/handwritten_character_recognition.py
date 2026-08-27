
"""
CodeAlpha Task 3 — Handwritten Character Recognition

This project recognizes handwritten digits using the MNIST dataset and a
Convolutional Neural Network (CNN).

Run:
    python handwritten_character_recognition.py

The script downloads MNIST through Keras, normalizes the images, trains a CNN,
evaluates it on the test set, saves a confusion matrix, training curves,
sample predictions, and the trained model.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

def load_and_prepare_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # CNN input shape: height x width x channel.
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return x_train, y_train, x_test, y_test

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.30),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def save_training_curves(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "accuracy_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "loss_curve.png", dpi=200)
    plt.close()

def save_sample_predictions(model, x_test, y_test):
    predictions = model.predict(x_test[:12], verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    plt.figure(figsize=(12, 7))
    for i in range(12):
        ax = plt.subplot(3, 4, i + 1)
        plt.imshow(x_test[i].squeeze(), cmap="gray")
        plt.title(f"True: {y_test[i]} | Pred: {predicted_labels[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "sample_predictions.png", dpi=200)
    plt.close()

def main():
    x_train, y_train, x_test, y_test = load_and_prepare_data()

    print("Training images:", x_train.shape)
    print("Test images:", x_test.shape)

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )
    ]

    history = model.fit(
        x_train, y_train,
        validation_split=0.10,
        epochs=15,
        batch_size=128,
        callbacks=callbacks,
        verbose=1
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    probabilities = model.predict(x_test, verbose=0)
    predicted_labels = np.argmax(probabilities, axis=1)

    print("\nClassification report:")
    print(classification_report(y_test, predicted_labels, digits=4))

    cm = confusion_matrix(y_test, predicted_labels)
    ConfusionMatrixDisplay(cm).plot()
    plt.title("MNIST CNN — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

    save_training_curves(history)
    save_sample_predictions(model, x_test, y_test)

    model.save(RESULTS_DIR / "mnist_cnn.keras")

    with open(RESULTS_DIR / "test_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Test loss: {test_loss:.6f}\n")
        f.write(f"Test accuracy: {test_accuracy:.6f}\n")

    print("\nSaved model and evaluation files in the results/ folder.")

if __name__ == "__main__":
    main()
