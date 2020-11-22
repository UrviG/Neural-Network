import numpy as np
import pandas as pd

def preprocess(dataset):
    # Load dataset and scale all values between 0 and 1.
    df = pd.read_excel(dataset)
    df = (df - df.min()) / (df.max() - df.min())

    # Split dataset into training and testing set.
    train = df.sample(frac=0.8, random_state=200)
    test = df.drop(train.index)
    return train, test

class NeuralNet:
    def __init__(self, learning_rate, epochs, activation_function):
        self.lr = learning_rate
        self.num_iterations = epochs
        self.act = activation_function

    # Three Possible Activation Functions: ReLu, Sigmoid, and Tanh.
    def activation(self, x):
        if self.act == 'relu':
            return np.maximum(0,x)

        elif self.act == 'sigmoid':
            return 1 / (1 + np.exp(-x))

        elif self.act == 'tanh':
            return np.tanh(x)

    # Derivatives of activation functions.
    def derivative(self, x):
        if self.act == 'relu':
            x[x <= 0] = 0
            x[x > 0] = 1
            return x
        elif self.act == 'sigmoid':
            return x * (1-x)

        elif self.act == 'tanh':
            return (1 - np.tanh(x)**2)

    # Initialize weights.
    def init_weights(self, input):
        self.params = {}

        # 2 hidden layers: number of neurons in each.
        self.h1_n = 10
        self.h2_n = 5

        # Weights and biases for first hidden layer.
        self.params['W1'] = np.random.rand(input, self.h1_n)
        self.params['b1'] = np.random.rand(self.h1_n)


        # Weights and biases for second hidden layer
        self.params['W2'] = np.random.rand(self.h1_n, self.h2_n)
        self.params['b2'] = np.random.rand(self.h2_n)

        # Weights for output layer (1 node)
        self.params['W3'] = np.random.rand(self.h2_n, 1)
        self.params['b3'] = np.random.rand(1)

    # Train model on training set.
    def train(self, train_set):
        nrows, ncols = train_set.shape[0], train_set.shape[1]
        x = train_set.iloc[:, 0:(ncols - 1)].values.reshape(nrows, ncols - 1)
        y = train_set.iloc[:, (ncols - 1)].values.reshape(nrows, 1)

        # First, initialize weights to random numbers.
        self.init_weights(ncols-1)


        for i in range(self.num_iterations):
            # Forward pass
            h1 = np.dot(x, self.params['W1']) + self.params['b1']
            h1 = self.activation(h1)
            h2 = np.dot(h1, self.params['W2']) + self.params['b2']
            h2 = self.activation(h2)
            out = np.dot(h2, self.params['W3']) + self.params['b3']
            out = self.activation(out)

            # Calculate errors for output and hidden nodes.
            error_out = self.derivative(out) * (y - out)
            error_h2 = self.derivative(h2) * (np.transpose(self.params['W3']) * error_out)
            error_h1 = self.derivative(h1) * (np.sum(self.params['W2']*np.reshape(error_h2, (nrows,1,self.h2_n)), axis=2))

            # Update weights based on errors calculated previously.
            delta_w3 = self.lr * error_out * h2
            delta_w3 = np.sum(delta_w3, axis=0)
            delta_w3 = np.reshape(delta_w3, (self.h2_n,1))
            delta_w2 = self.lr * np.reshape(error_h2, (nrows,1,self.h2_n)) * np.reshape(h1, (nrows,self.h1_n,1))
            delta_w2 = np.sum(delta_w2, axis=0)
            delta_w1 = self.lr * np.reshape(error_h1, (nrows,1,self.h1_n)) * np.reshape(x, (nrows,ncols-1,1))
            delta_w1 = np.sum(delta_w1, axis=0)

            self.params['W3'] += delta_w3
            self.params['W2'] += delta_w2
            self.params['W1'] += delta_w1

            # Update bias weights
            delta_b3 = self.lr * error_out
            delta_b3 = np.sum(delta_b3, axis=0)
            delta_b2 = self.lr * error_h2
            delta_b2 = np.sum(delta_b2, axis=0)
            delta_b1 = self.lr * error_h1
            delta_b1 = np.sum(delta_b1, axis=0)

            self.params['b3'] += delta_b3
            self.params['b2'] += delta_b2
            self.params['b1'] += delta_b1

        # Compute and return training accuracy.
        return self.test(train_set)

    # Test model on test set.
    def test(self, test_set):
        nrows, ncols = test_set.shape[0], test_set.shape[1]
        x = test_set.iloc[:, 0:(ncols - 1)].values.reshape(nrows, ncols - 1)
        y = test_set.iloc[:, (ncols - 1)].values.reshape(nrows, 1)

        # Forward pass.
        h1 = np.dot(x, self.params['W1']) + self.params['b1']
        h1 = self.activation(h1)
        h2 = np.dot(h1, self.params['W2']) + self.params['b2']
        h2 = self.activation(h2)
        out = np.dot(h2, self.params['W3']) + self.params['b3']
        out = self.activation(out)

        # Calculate and return testing accuracy.
        out[out >= 0.5] = 1
        out[out < 0.5] = 0
        return (out == y).all(axis=1).mean()




if __name__ == "__main__":
    # Preprocess data.
    training_set, testing_set = preprocess("Cryotherapy.xlsx")

    # initialize neural net with parameters
    # best: h1 = 10 & h2 = 5
    #       learning_rate=0.01, epochs=500, activation_function='tanh'
    nn = NeuralNet(learning_rate=0.01, epochs=5000, activation_function='tanh')
    train_acc = nn.train(training_set)
    print("\nTraining Accuracy: {:0.2f}\n".format(train_acc*100))
    test_acc = nn.test(testing_set)
    print("\nTesting Accuracy: {:0.2f}\n".format(test_acc * 100))