import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


train_dataframe = pd.read_csv('data/train.csv')
train_data = np.array(train_dataframe)

# m = number of cases to train our model
# n = size of test case (image_size + label)
m, n = train_data.shape
k = 1000 # number of test cases

test_data = train_data[0:k].T
test_labels = test_data[0]
test_images = test_data[1:n] / 255.

train_data = train_data[k:m].T
train_labels = train_data[0]
train_images = train_data[1:n] / 255.

def init_parameters():
    W1 = np.random.rand(10,784) - 0.5 # generates a [10,784] array of numbers in [-0.5, 0.5]
    b1 = np.random.rand(10,1) - 0.5 
    W2 = np.random.rand(10,10) - 0.5
    b2 = np.random.rand(10,1) - 0.5 
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z,0)

def ReLU_deriv(Z):
    return Z > 0

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max()+1)) # zeros only
    # Y.size = number of test cases
    # Y.max = 9 (maximum of values in Y) this only works with numerical categories starting with zero
    one_hot_Y[np.arange(Y.size), Y] = 1
    # first index = 0, 1, ..., Y.size-1
    # second index = labels
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1/m * dZ2.dot(A1.T)
    db2 = 1/m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1/m * dZ1.dot(X.T)
    db1 = 1/m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def show_accuracy(i, A2, Y):
    print("Iteration: ", i)
    predictions = get_predictions(A2)
    print(get_accuracy(predictions, Y))

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_parameters()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            show_accuracy(i, A2, Y)
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(train_images, train_labels, 0.1, 500)

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = train_data[:, index, None]
    prediction = make_predictions(current_image, W1, b1, W2, b2)
    label = train_labels[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((28,28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

def accuracy_per_label(Y, predictions, train_labels):
    labels = np.unique(Y)

    accuracies = []
    quantities = []
    for label in labels:
        # Iterate through possible labels
        label_mask = Y == label
        total = np.sum(label_mask)
        correct = np.sum(predictions[label_mask] == label)
        accuracy = correct / total if total > 0 else 0
        accuracies.append(accuracy)
        quantities.append(np.sum(train_labels == label))

    # Accuracy
    plt.figure(figsize=(8, 5))
    plt.bar(labels, accuracies, color='skyblue')
    plt.xlabel('Digit')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per digit')
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.xticks(labels)
    plt.show()

    # Database distribution
    plt.figure(figsize=(8, 5))
    plt.bar(labels, quantities, color='skyblue')
    plt.xlabel('Digit')
    plt.ylabel('Appereances')
    plt.title('Database appereances per digit')
    plt.grid(axis='y')
    plt.xticks(labels)
    plt.show()


test_predictions = make_predictions(test_images, W1, b1, W2, b2)
print(get_accuracy(test_predictions, test_labels))
accuracy_per_label(test_labels, test_predictions, train_labels)