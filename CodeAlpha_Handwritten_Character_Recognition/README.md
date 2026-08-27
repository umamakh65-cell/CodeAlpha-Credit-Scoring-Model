# CodeAlpha Task 3 — Handwritten Character Recognition

## Project overview

This project identifies handwritten digits using a Convolutional Neural Network, commonly called a CNN. The model is trained on the MNIST dataset, where every example is a 28×28 grayscale image containing one handwritten digit from 0 through 9. The complete workflow loads the dataset, normalizes the pixel values, reshapes the images for CNN input, builds the neural network, trains it with a validation split, evaluates it on unseen test data, produces a classification report and confusion matrix, visualizes training progress, displays sample predictions, and saves the trained model.

## Dataset

MNIST is a standard benchmark dataset for handwritten digit recognition. It contains 60,000 training images and 10,000 test images. Every image is 28 pixels wide by 28 pixels high and has a label from 0 to 9. TensorFlow/Keras provides a convenient dataset loader, so the project can download the dataset automatically when `tf.keras.datasets.mnist.load_data()` is first called.

## Preprocessing

The original MNIST pixel values are integers from 0 to 255. The project converts them to floating-point values and divides by 255 so that each pixel lies between 0 and 1. Neural networks generally train more smoothly when inputs are scaled to a small, consistent range. A CNN expects an explicit channel dimension, so each image is reshaped from `(28, 28)` to `(28, 28, 1)`, where the final `1` represents the grayscale channel.

## CNN architecture

The first convolutional layer contains 32 filters. Convolutional filters learn local visual patterns such as edges and curves. A ReLU activation introduces nonlinearity so the network can learn more complex relationships. Max pooling reduces the spatial dimensions while retaining strong features. A second convolutional layer with 64 filters learns higher-level patterns from the features produced by the first layer. The feature maps are then flattened and passed to a dense layer with 128 neurons. Dropout randomly removes a fraction of training activations to reduce overfitting. The final dense layer contains 10 neurons, one for each digit, and uses softmax to produce probabilities whose total is 1.

## Training

The network uses the Adam optimizer, which adapts the learning rate during training and is a strong general-purpose optimizer for neural networks. Sparse categorical cross-entropy is used because the labels are integer class IDs from 0 to 9. Ten percent of the training data is used as validation data, allowing us to monitor how well the network generalizes while it trains. Early stopping watches validation loss and restores the best weights when validation performance stops improving.

## Evaluation

The final evaluation is performed on the separate 10,000-image MNIST test set. Test accuracy reports the proportion of digits classified correctly. The classification report provides precision, recall, and F1-score for each digit class. The confusion matrix shows which digits are most frequently confused with one another. This is especially useful because visually similar handwritten digits can be harder for the model to distinguish.

## Training curves

The project saves accuracy and loss curves for both training and validation data. If training accuracy keeps increasing while validation accuracy stops improving or decreases, that can indicate overfitting. If both improve together, the model is generally learning useful patterns. These plots make the training process easier to explain in the internship presentation.

## Sample predictions

The project also saves a grid containing test images with their true labels and the CNN's predicted labels. This gives a direct visual demonstration of the system's output and is useful for the GitHub repository and LinkedIn project explanation.

## Output

Running the script creates a `results/` folder containing the trained `mnist_cnn.keras` model, accuracy and loss plots, a confusion matrix, sample predictions, and a text file containing the final test loss and accuracy.

## How to run

Install the dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python handwritten_character_recognition.py
```

The first run needs internet access so Keras can download MNIST.

## Extension

The assignment mentions that the task can be extended to full word or sentence recognition using sequence models such as CRNN. The submitted version intentionally focuses on the MNIST digit-recognition problem because it directly satisfies the handwritten-character recognition objective while keeping the project reproducible and understandable.
