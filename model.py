import torch
from torch import nn
from torch.nn import *

class SimpleNeuralNetwork(nn.Module):
    def __init__(self,input_size, hidden_size, class_count):
        super().__init__()
        print("hidden layer size: {}".format(hidden_size))
        torch.manual_seed(1)
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
        input_data = torch.tanh(input_data)
        input_data = self.linear_layer2(input_data)
        input_data = torch.sigmoid(input_data)
        return input_data

# linear->sigmoid->linear->sigmoid 200 epoch: 0.1928274929523468 5th
# linear->tanh->linear->sigmoid 200 epoch (hidden layer size 2048): 0.1775105595588684 2nd
# linear->tanh->linear->sigmoid 200 epoch (hidden layer size input_size/16): 0.17838864028453827 3rd
# linear->tanh->linear->sigmoid 200 epoch (hidden layer size input_size/8 (3664)): 0.17661263048648834  1st (stops improving at epoch 174 at a loss of 0.17529436945915222)
# linear->tanh->linear->sigmoid 200 epoch (hidden layer size input_size/32): 0.18867142498493195 4th

# Identical to above first place but the data was modified to no longer include punctuation/symbols. Stops improving at epoch 177 with a loss of 0.17445825040340424 
