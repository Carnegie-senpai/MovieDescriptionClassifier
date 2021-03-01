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

training_key_file = open("./data/training_keys","rb")
verifying_key_file = open("./data/verifying_keys","rb")
testing_key_file = open("./data/testing_keys","rb")

testing_data_file = open("./data/testing_data_tensor","rb")
testing_target_file = open("./data/testing_targets","rb")

classes:list = sorted(list(load(classes_file)))
raw_data = load(raw_file)

print(classes)
training_data = load(training_file)
training_targets = load(training_targets_file)
training_keys = load(training_key_file)
verifying_keys = load(verifying_key_file)
testing_keys = load(testing_key_file)

testing_data = load(testing_data_file)
testing_targets = load(testing_target_file)

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
    loss = BCELoss().cuda()
    optimizer = optim.Adam(params=model.parameters(),lr=learning_rate)
    for i in range(epochs):
        model_output = model(data)
        loss_amount = loss(model_output.float(),training_targets.float())
        print("epoch[{}]: {}".format(i,loss_amount.item()))
        print("prediction: ",determine_genres(model_output[0]))
        print("truth:",determine_genres(training_targets[1199]))
        print(training_keys[1199])
        model.zero_grad()
        loss_amount.backward()
        optimizer.step()
    return model

test_loss = BCELoss().cuda()
model = classifier(training_data,.0005,1)
testing_out = model(testing_data)
# testing_loss = test_loss(testing_out.float(),testing_targets.float())
# print("testing loss: ",testing_loss.item())
# out_file = open("./data/trained_model_output","wb")
# dump(model,out_file)