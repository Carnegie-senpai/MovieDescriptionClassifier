from sys import argv
from pickle import load,dump
from collections import defaultdict
'''
Creates a trigram bag of words based on the descriptions of the movie_dict
'''
if (len(argv) != 2 ):
    raise Exception("Expected 1 argument received {}".format(len(argv)-1)) 
file = open(argv[1],"rb")

movie_dict = load(file)

def trigram(index:int,array: list,):
    if index+2 < len(array):
        return "{}{}{}".format(array[index].strip(".,;:'\"?!"),array[index+1].strip(".,;:'\"?!"),array[index+2].strip(".,;:'\"?!"))
    return ""

bag_of_words = defaultdict(int)
for key in movie_dict.keys():
    temp_sentence =  movie_dict[key]["description"].split(" ")
    for i in range(len(temp_sentence)):
        if trigram(i,temp_sentence) != "":
            bag_of_words[trigram(i,temp_sentence)]+=1 

outfile = open("./data/pickled_bag_of_words","wb")

dump(bag_of_words,outfile)