from sklearn.feature_extraction.text import TfidfVectorizer
from model import SimpleNeuralNetwork
from pickle import load, dump
import torch
import numpy as np
from torch import optim

training_file = open("./data/training_data_tensor","rb")
raw_file = open("./data/pickled_data","rb")
classes_file = open("./data/pickled_genres","rb")
classes:set = load(classes_file)
raw_data = load(raw_file)
print(raw_data["genres"])
training_data = load(training_file)
assert type(training_data) == torch.Tensor

def classifier(data:torch.Tensor,learning_rate:float,epochs:int):
    print(data.__len__())
    model = SimpleNeuralNetwork(len(data),int(len(data)/2),len(classes))
    optimizer = optim.Adam(params=model,lr=learning_rate)
    for i in range(epochs):
        #run model on data
        #compare loss based on model's value and actual value
        #backpropogate loss
        #zero gradients
        #step optimizer

classifier(training_data,.0001,2000)