from sys import argv
from pickle import load,dump
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from random import shuffle
'''
Creates a trigram bag of words based on the descriptions of the movie_dict. Note, when run it will override pickled_bag_of_words with new randomly selected values
'''
if (len(argv) != 2 ):
    raise Exception("Expected 1 argument received {}".format(len(argv)-1)) 
file = open(argv[1],"rb")

movie_dict = load(file)

# def trigram(index:int,array: list,):
#     if index+2 < len(array):
#         return "{}{}{}".format(array[index].strip(".,;:'\"?!"),array[index+1].strip(".,;:'\"?!"),array[index+2].strip(".,;:'\"?!"))
#     return ""
print("Creating Bag of Words")
bag_of_words = defaultdict(int)
random_list = list(range(0,len(movie_dict.keys())))
shuffle(random_list)
training_size = int(.6*len(random_list))
verifying_size = int(.8*len(random_list))
k = list(movie_dict.keys())

training_data = []
verifying_data = []
testing_data = []
print("Selecting Training Data")
for index in range(0,training_size):
    training_data.append(movie_dict[k[random_list[index]]]["description"])
print("Selecting Verification Data")
for index in range(training_size,verifying_size):
    verifying_data.append(movie_dict[k[random_list[index]]]["description"])
print("Selecting Testing Data")
for index in range(verifying_size,len(random_list)):
    testing_data.append(movie_dict[k[random_list[index]]]["description"])

print("Vectorizing Training Data")
training_vector = TfidfVectorizer(ngram_range=(1,3))
print("Vectorizing Verification Data")
verifying_vector = TfidfVectorizer(ngram_range=(1,3))
print("Vectorizing Testing Data")
testing_vector = TfidfVectorizer(ngram_range=(1,3))

print("Determining idf and vocabulary for Training Data")
training_vector.fit_transform(training_data)
print("Determining idf and vocabulary for Verification Data")
verifying_vector.fit_transform(verifying_data)
print("Determining idf and vocabulary for Testing Data")
t= testing_vector.fit_transform(testing_data)

print ("Writing Results to Files")
training_file = open("./data/training_data_vector","wb")
verifying_file = open("./data/verification_data_vector","wb")
testing_file = open("./data/testing_data_vector","wb")
dump(training_vector,training_file)
dump(verifying_vector,verifying_file)
dump(testing_vector,testing_file)
print("Finished Creating Bag of Words")