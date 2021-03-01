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
        self.linear_layer = Linear(input_size,hidden_size).to(device=self.device)
        self.linear_layer2 = Linear(hidden_size,class_count).to(device=self.device)


    def forward(self,input_data:torch.Tensor):
        input_data = self.linear_layer(input_data)
        input_data = torch.sigmoid(input_data)
        input_data = self.linear_layer2(input_data)
        input_data = torch.sigmoid(input_data)
        return input_data
