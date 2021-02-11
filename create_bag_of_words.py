from sys import argv
from pickle import load, dump
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from random import shuffle
import numpy as np
import torch
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

# def trigram(index:int,array: list,):
#     if index+2 < len(array):
#         return "{}{}{}".format(array[index].strip(".,;:'\"?!"),array[index+1].strip(".,;:'\"?!"),array[index+2].strip(".,;:'\"?!"))
#     return ""
print("Creating Bag of Words")
bag_of_words = defaultdict(int)
random_list = list(range(0, len(movie_dict.keys())))
shuffle(random_list)
training_size = int(.6*len(random_list))
verifying_size = int(.8*len(random_list))
k = list(movie_dict.keys())

training_data = []
verifying_data = []
testing_data = []
vprint("Selecting Training Data")
for index in range(0, training_size):
    training_data.append(movie_dict[k[random_list[index]]]["description"])
vprint("Selecting Verification Data")
for index in range(training_size, verifying_size):
    verifying_data.append(movie_dict[k[random_list[index]]]["description"])
vprint("Selecting Testing Data")
for index in range(verifying_size, len(random_list)):
    testing_data.append(movie_dict[k[random_list[index]]]["description"])

vprint("Vectorizing Training Data")
training_vector = TfidfVectorizer(ngram_range=(1, 3))
vprint("Vectorizing Verification Data")
verifying_vector = TfidfVectorizer(ngram_range=(1, 3))
vprint("Vectorizing Testing Data")
testing_vector = TfidfVectorizer(ngram_range=(1, 3))

vprint("Constructing tfidf sparse matrix for training data")
training_tfidf = training_vector.fit_transform(training_data)
vprint("Constructing tfidf sparse matrix for verification data")
verifying_tfidf = verifying_vector.fit_transform(verifying_data)
vprint("Constructing tfidf sparse matrix for testing data")
testing_tfidf = testing_vector.fit_transform(testing_data)

training_data = []
verifying_data = []
testing_data = []


def convert_to_tensor(sparse_csr_matrix):
    sparse_coo = sparse_csr_matrix.tocoo().astype(np.float32)
    if torch.cuda.is_available():
        device = "cuda"
        indices = torch.from_numpy(np.vstack((sparse_coo.row, sparse_coo.col))).long().cuda()
        values = torch.from_numpy(sparse_coo.data).cuda()
    else:
        indices = torch.from_numpy(np.vstack((sparse_coo.row, sparse_coo.col))).long()
        values = torch.from_numpy(sparse_coo.data)
        device = "cpu"
    shape = torch.Size(sparse_coo.shape)
    return torch.sparse_coo_tensor(indices, values, shape, device=torch.device(device))


vprint("Converting training tfidf matrix to tensor")
training_tensor = convert_to_tensor(training_tfidf)
vprint("Converting verifying tfidf matrix to tensor")
verifying_tensor = convert_to_tensor(verifying_tfidf)
vprint("Converting testing tfidf matrix to tensor")
testing_tensor = convert_to_tensor(testing_tfidf)

vprint("Writing Results to Files")
training_file = open("./data/training_data_tensor", "wb")
verifying_file = open("./data/verification_data_tensor", "wb")
testing_file = open("./data/testing_data_tensor", "wb")

training_vectorizer_file = open("./data/training_data_vectorizer","wb")
verifiying_vectorizer_file = open("./data/verifiying_data_vectorizer","wb")
testing_vectorizer_file = open("./data/testing_data_vectorizer","wb")


dump(training_tensor, training_file)
dump(verifying_tensor, verifying_file)
dump(testing_tensor, testing_file)

dump(training_tfidf,training_vectorizer_file)
dump(verifying_tfidf,verifiying_vectorizer_file)
dump(testing_tfidf,testing_vectorizer_file)

print("Finished Creating Bag of Words")
