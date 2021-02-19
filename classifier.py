from sklearn.feature_extraction.text import TfidfVectorizer
from model import SimpleNeuralNetwork
from pickle import load, dump
import torch
import numpy as np
from torch import optim
from torch.nn import BCELoss

training_file = open("./data/training_data_tensor","rb")
raw_file = open("./data/pickled_data","rb")
classes_file = open("./data/pickled_genres","rb")
training_targets_file = open("./data/training_targets","rb")
classes:list = sorted(list(load(classes_file)))
raw_data = load(raw_file)
print(classes)
training_data = load(training_file)
training_targets = load(training_targets_file)
#print("training data: ",training_data.shape)
#print("training targets: ",training_targets.shape)
assert type(training_data) == torch.Tensor
assert type(training_targets) == torch.Tensor

def determine_genres(prediction:torch.Tensor ):
    result = []
    debug = []
    for i in range(len(classes)):
        if prediction[i].item() > .9:
            result.append(classes[i])
        debug.append(prediction[i].item())
    #print(debug)
    return result

def classifier(data:torch.Tensor,learning_rate:float,epochs:int):
    global training_targets
    model = SimpleNeuralNetwork(data.shape[1],2048,len(classes))
    optimizer = optim.Adam(params=model.parameters(),lr=learning_rate)
    loss = BCELoss().cuda()
    for i in range(epochs):
        #run model on data
        model_output = model(data)
    #    print("model output: ",model_output)
    #    print("model output shape: ",model_output.shape)
    #    print("training targets: ",training_targets)
    #    print("training targets shape",training_targets.shape)
        loss_amount = loss(model_output.float(),training_targets.float())
        print("epoch[{}]: {}".format(i,loss_amount.item()))
        print("prediction: ",determine_genres(model_output[0]))
        print("truth:",determine_genres(training_targets[0]))
        model.zero_grad()
        loss_amount.backward()
        optimizer.step()
        #compare loss based on model's value and actual value
        #backpropogate loss
        #zero gradients
        #step optimizer

classifier(training_data,.0005,1000)