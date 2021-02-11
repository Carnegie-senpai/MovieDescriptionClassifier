import torch
from torch import nn
from torch.nn import CrossEntropyLoss, Conv1d,Linear

class SimpleNeuralNetwork(nn.Module):
    def __init__(self,input_size, hidden_size, class_count):
        super().__init__()
        torch.random.seed()
        if torch.cuda.is_available():
            self.device = "cuda"
            self.cuda()
        else:
            self.device = "cpu"
            self.cpu()
        self.loss = CrossEntropyLoss()
        self.linear_layer = Linear(input_size,hidden_size)
        self.linear_layer2 = Linear(hidden_size,class_count)


    def forward(self,input_data):
        # Input is a vectorized set of tfidf values for a given document
        # This is transformed into a probability value using softmax

        # dot product of weight for input value
        # get softmax of product
        #return
        input_data = self.linear_layer(input_data)
        input_data = nn.functional.sigmoid(input_data)
        input_data = self.linear_layer2(input_data)
        return input_data
