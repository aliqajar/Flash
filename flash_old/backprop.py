

import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x)) 

def sigmoid_derivative(x):
    return x * (1-x)

n_inputs = 2
n_hidden = 3
n_outputs = 1

np.random.seed(42)
weights_input_hidden = np.random.rand(n_inputs, n_hidden)
weights_hidden_output = np.random.rand(n_hidden, n_outputs)
bias_hidden = np.random.rand(n_hidden)
bias_output = np.random.rand(n_outputs)

def forward_pass(inputs):

    hidden_layer_input = np.dot(inputs, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    final_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    final_layer_output = sigmoid(final_layer_input)
    return hidden_layer_output, final_layer_output


def backpropagation(inputs, hidden_layer_output, predicted_output, actual_output):

    error = actual_output - predicted_output
    d_predicted_output =  error * sigmoid_derivative(predicted_output)

    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    weights_hidden_output_gradient = hidden_layer_output.T.dot(d_predicted_output)
    weights_input_hidden_gradient = inputs.T.dot(d_hidden_layer)
    bias_hidden_gradient = np.sum(d_hidden_layer, axis=0, keepdims=True)
    bias_output_gradient = np.sum(d_predicted_output, axis=0, keepdims=True)

    return weights_input_hidden_gradient, weights_hidden_output_gradient, bias_hidden_gradient, bias_output_gradient


inputs = np.array([[0.5, 0.2]])
actual_output = np.array([[1]])

hidden_layer_output, predicted_output = forward_pass(inputs)
weights_input_hidden_gradient, weights_hidden_output_gradient, bias_hidden_gradient, bias_output_gradient = backpropagation(inputs, hidden_layer_output, predicted_output, actual_output)

print(hidden_layer_output)



