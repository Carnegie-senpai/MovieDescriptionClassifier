from sklearn.feature_extraction.text import TfidfVectorizer
from model import SimpleNeuralNetwork
from pickle import load, dump
import torch
import numpy as np
global device 
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
vector_file = open("./training_vector_file","rb")
model_file = open("./trained_model_output","rb")
vector = load(vector_file)
model = load(model_file)

classes_file = open("./data/pickled_genres","rb")
classes:list = sorted(list(load(classes_file)))

def convert_to_tensor(sparse_csr_matrix):
    sparse_coo = sparse_csr_matrix.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_coo.row, sparse_coo.col))).long().to(device=device)
    values = torch.from_numpy(sparse_coo.data).to(device=device)
    shape = torch.Size(sparse_coo.shape)
    return torch.sparse_coo_tensor(indices, values, shape, device=torch.device(device))

def string_to_vector(input_text:str):
    np_arr = vector.transform([input_text])
    return convert_to_tensor(np_arr)

def determine_genres(prediction:torch.Tensor ):
    result = []
    debug = []
    for i in range(len(classes)):
        if prediction[i].item() > .5:
            result.append(classes[i])
        debug.append(prediction[i].item())
    return result

def run_model(input_text:str):
    in_tensor = string_to_vector(input_text)
    out_tensor = model(in_tensor)
    print(input_text,": ",determine_genres(out_tensor[0]))
    return out_tensor[0]
if __name__ == "__main__":
    run_model("A meek Hobbit from the Shire and eight companions set out on a journey to destroy the powerful One Ring and save Middle-earth from the Dark Lord Sauron.")