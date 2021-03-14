from sklearn.feature_extraction.text import TfidfVectorizer
from model import SimpleNeuralNetwork
from pickle import load, dump
import torch
import numpy as np
from torch import optim
from torch.nn import BCELoss
from random import randrange
global device 
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
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
        if prediction[i].item() > .5:
            result.append(classes[i])
        debug.append(prediction[i].item())
    #print(debug)
    return result



def classifier(data:torch.Tensor,learning_rate:float,epoch:int=0):
    global training_targets
    model = SimpleNeuralNetwork(data.shape[1],data.shape[1]//8,len(classes))
    loss = BCELoss().to(device=device)
    test_loss = BCELoss().to(device=device)
    
    optimizer = optim.Adam(params=model.parameters(),lr=learning_rate)


    one_loss_back = 100
    two_loss_back = 1000
    i = 0
    if (epoch == 0):
        while one_loss_back < two_loss_back:
        # for i in range(epoch):
            model_output = model(data)
            loss_amount = loss(model_output.float(),training_targets.float())

            testing_out = model(testing_data)
            testing_loss = test_loss(testing_out.float(),testing_targets.float())
            if (i%5==0):
                print("---epoch {}---".format(i))
                rand = randrange(0,data.shape[1])
                print("testing loss: ",testing_loss.item())
                print("prediction: ",determine_genres(model_output[rand]))
                print("truth:",determine_genres(training_targets[rand]))
                print("raw output: ",model_output[rand])
                print("genres: ",classes)
            
            model.zero_grad()
            loss_amount.backward()
            optimizer.step()
            i+=1
            two_loss_back = one_loss_back
            one_loss_back = testing_loss.item()
        print("Converged at epoch {} at loss {}".format(i-1,one_loss_back))
    else:
        for i in range(epoch):
            model_output = model(data)
            loss_amount = loss(model_output.float(),training_targets.float())

            testing_out = model(testing_data)
            testing_loss = test_loss(testing_out.float(),testing_targets.float())
            if (i%5==0):
                print("---epoch {}---".format(i))
                print("testing loss: ",testing_loss.item())
                print("prediction: ",determine_genres(model_output[140]))
                print("truth:",determine_genres(training_targets[140]))
            
            model.zero_grad()
            loss_amount.backward()
            optimizer.step()
    return model

if __name__ == "__main__":
    model = classifier(training_data,.0005)

    out_file = open("./trained_model_output","wb")
    dump(model,out_file)

