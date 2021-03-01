from sys import argv
from pickle import load, dump
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from random import shuffle
import numpy as np
import torch

global device 
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
# Creates the training data sets. Borken up into different data set based on usage.
# 60% is dedicated to training. 20% to verification. 20% to testing. All data is
# vectorized bag of words with tfidf. After data is finished being processed it is pickled
# and written to files in the data folder.
verbose = False
if (len(argv) != 2):
    if len(argv) == 3 and (argv[2] == "--verbose" or argv[2] == "-v"):
        verbose = True
    else:
        raise Exception("Expected 1 argument received {}".format(len(argv)-1))
file = open(argv[1], "rb")
def vprint(msg):
    global verbose
    if (verbose):
        print(msg)
movie_dict = load(file)
genres:list = sorted(list(load(open("./data/pickled_genres","rb"))))

print("Creating Bag of Words")
bag_of_words = defaultdict(int)
random_list = list(range(0, len(movie_dict.keys())))
shuffle(random_list)
training_size = int(.6*len(random_list))
verifying_size = int(.8*len(random_list))
k = list(movie_dict.keys())

training_data = []
training_target = []
verifying_data = []
verifying_target = []
testing_data = []
testing_target = []
training_keys = []
verifying_keys = []
testing_keys = []


# This code block randomly creates training, testing, and verifying data out of 
# the data set. This is to try and account for any biases that may arrise from the
# ordering of the dataset. 
vprint("Selecting Training Data")
for index in range(0, training_size):
    training_data.append(movie_dict[k[random_list[index]]]["description"])
    training_target.append(movie_dict[k[random_list[index]]]["genre"])
    training_keys.append(k[random_list[index]])
print(k[random_list[1]],movie_dict[k[random_list[1]]]["genre"],training_data[1])
vprint("Selecting Verification Data")
for index in range(training_size, verifying_size):
    verifying_data.append(movie_dict[k[random_list[index]]]["description"])
    verifying_target.append(movie_dict[k[random_list[index]]]["genre"])
    verifying_keys.append(k[random_list[index]])
vprint("Selecting Testing Data")
for index in range(verifying_size, len(random_list)):
    testing_data.append(movie_dict[k[random_list[index]]]["description"])
    testing_target.append(movie_dict[k[random_list[index]]]["genre"])
    testing_keys.append(k[random_list[index]])


# Creates training vector
vprint("Vectorizing Training Data")
training_vector = TfidfVectorizer(ngram_range=(1, 3),max_df=0.8,min_df=10)
# vprint("Vectorizing Verification Data")
# verifying_vector = TfidfVectorizer(ngram_range=(1, 3),max_df=0.8,min_df=10)
# vprint("Vectorizing Testing Data")
# testing_vector = TfidfVectorizer(ngram_range=(1, 3),max_df=0.8,min_df=10)

# Creates the training sparse matrix
vprint("Constructing tfidf sparse matrix for training data")
training_sparse_matrix = training_vector.fit_transform(training_data)
testing_sparse_matrix = training_vector.transform(testing_data)
verifying_sparse_matrix = training_vector.transform(verifying_data)
# vprint("Constructing tfidf sparse matrix for verification data")
# verifying_tfidf = verifying_vector.fit_transform(verifying_data)
# vprint("Constructing tfidf sparse matrix for testing data")
# testing_tfidf = testing_vector.fit_transform(testing_data)
# print("params: ",training_vector.transform(testing_data) 
training_key_file = open("./data/training_keys","wb")
verifying_key_file = open("./data/verifying_keys","wb")
testing_key_file = open("./data/testing_keys","wb")

dump(training_keys,training_key_file)
dump(verifying_keys,verifying_key_file)
dump(testing_keys,testing_key_file)


def convert_to_tensor(sparse_csr_matrix):
    sparse_coo = sparse_csr_matrix.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_coo.row, sparse_coo.col))).long().to(device=device)
    values = torch.from_numpy(sparse_coo.data).to(device=device)
    shape = torch.Size(sparse_coo.shape)
    return torch.sparse_coo_tensor(indices, values, shape, device=torch.device(device))


vprint("Converting training tfidf matrix to tensor")
training_tensor = convert_to_tensor(training_sparse_matrix)
vprint("Converting verifying tfidf matrix to tensor")
verifying_tensor = convert_to_tensor(verifying_sparse_matrix)
vprint("Converting testing tfidf matrix to tensor")
testing_tensor = convert_to_tensor(testing_sparse_matrix)

vprint("Writing Results to Files")
training_file = open("./data/training_data_tensor", "wb")
verifying_file = open("./data/verification_data_tensor", "wb")
testing_file = open("./data/testing_data_tensor", "wb")

training_sparse_matrix_file = open("./data/training_sparse_matrix","wb")
verifiying_vectorizer_file = open("./data/verifiying_data_vectorizer","wb")
testing_vectorizer_file = open("./data/testing_data_vectorizer","wb")


dump(training_tensor, training_file)
dump(verifying_tensor, verifying_file)
dump(testing_tensor, testing_file)

# dump(training_sparse_matrix,training_sparse_matrix_file)
# dump(verifying_tfidf,verifiying_vectorizer_file)
# dump(testing_tfidf,testing_vectorizer_file)

print("Finished Creating Bag of Words")

print("Creating target tensor")
training_targets = []
verifying_targets = []
testing_targets = []
for value in training_target:
    vector = [0]*len(genres)
    for index in range(len(genres)):
        if genres[index] in value:
            vector[index] = 1
    training_targets.append(vector)
for value in verifying_target:
    vector = [0]*len(genres)
    for index in range(len(genres)):
        if genres[index] in value:
            vector[index] = 1
    verifying_targets.append(vector)
for value in testing_target:
    vector = [0]*len(genres)
    for index in range(len(genres)):
        if genres[index] in value:
            vector[index] = 1
    testing_targets.append(vector)



def targets_to_tensor(targets:list):
    tens = torch.from_numpy(np.array(targets)).long()
    return tens.to(device=device)

training_targets_file = (open("./data/training_targets","wb"))
verifying_targets_file = (open("./data/verifying_targets","wb"))
testing_targets_file = (open("./data/testing_targets","wb"))

training_target_tensor = targets_to_tensor(training_targets)
verifying_target_tensor = targets_to_tensor(verifying_targets)
testing_target_tensor = targets_to_tensor(testing_targets)


dump(training_target_tensor,training_targets_file)
dump(verifying_target_tensor,verifying_targets_file)
dump(testing_target_tensor,testing_targets_file)
print("Created target tensor")
