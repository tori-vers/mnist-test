import numpy as np
import mnist_loader

def load_data_wrapper():
    tr_d, va_d, te_d = mnist_loader.load_data_wrapper()
    training_data = [(np.reshape(x, (784, 1)), y) for x, y in tr_d]
    validation_data = [(np.reshape(x, (784, 1)), y) for x, y in va_d]
    test_data = [(np.reshape(x, (784, 1)), y) for x, y in te_d]
    return training_data, validation_data, test_data

# Load data
training_data, validation_data, test_data = load_data_wrapper()

# Create the network
net = Network([784, 30, 10])

# Train the network
net.SGD(training_data, 20, 10, 3.0, test_data=test_data)

# Code to display the first incorrectly classified image for each class after the last epoch
misclassifications, _ = net.evaluate_per_class(test_data)
first_misclassified_images = {}
for label, count in misclassifications.items():
    if count > 0:
        for x, y in test_data:
            predicted_label = np.argmax(net.feedforward(x))
            if predicted_label != y and y == label:
                first_misclassified_images[label] = x.reshape(28, 28)
                break

# Display the first misclassified images
# You need to implement a function to display multiple images in a single display window
# Example:
# display_multiple_images(first_misclassified_images)
