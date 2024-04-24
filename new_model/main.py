import torch
import torch.nn as nn

class FCLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(FCLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

class SoftmaxLayer(nn.Module):
    def __init__(self, dim=1):
        super(SoftmaxLayer, self).__init__()
        self.softmax = nn.Softmax(dim=dim)

    def forward(self, x):
        return self.softmax(x)

class ElementwiseMultiplication(nn.Module):
    def __init__(self):
        super(ElementwiseMultiplication, self).__init__()

    def forward(self, x1, x2):
        return x1 * x2

class FCBlockWithGELU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FCBlockWithGELU, self).__init__()
        self.fc_input = FCLayer(input_size, hidden_size)
        self.activation = nn.GELU()
        self.fc_output = FCLayer(hidden_size, output_size)

    def forward(self, x):
        x = self.fc_input(x)
        x = self.activation(x)
        x = self.fc_output(x)
        return x

class TestModel(nn.Module):
    def __init__(self, input_size, hidden_size, gelu_size):
        super(TestModel, self).__init__()
        self.fc1 = FCLayer(input_size, hidden_size)
        self.fc2 = FCLayer(input_size, hidden_size)
        self.softmax = SoftmaxLayer(dim=1)
        self.multiply = ElementwiseMultiplication()
        self.fc_block = FCBlockWithGELU(hidden_size, gelu_size, hidden_size)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x1_softmax = self.softmax(x1)
        x2 = self.fc_block(x2)
        output = self.multiply(x1_softmax, x2)
        return output