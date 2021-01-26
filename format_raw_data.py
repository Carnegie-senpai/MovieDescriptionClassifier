
import csv
from sys import argv
from pickle import dump, load
print("Opening csv file")


csvfile = open(argv[1], encoding="utf-8")
count = 0
movie_list = []
print("Reading csv file")
for row in csv.reader(csvfile):
    temp = []
    for item in row:
        temp.append(item.lower().strip())
    movie_list.append(temp)

print("Gather genres")
genres = set()
for row in movie_list:
    for item in row[5].split(","):
        genres.add(item.strip())

    movie_dict = {}

print("Creating dictionary")
for row in movie_list[1:]:
    movie_dict[row[0]] = {
        movie_list[0][1]:row[1],
        movie_list[0][2]:row[2],
        movie_list[0][3]:row[3],
        movie_list[0][4]:row[4],
        movie_list[0][5]:row[5],
        movie_list[0][6]:row[6],
        movie_list[0][7]:row[7],
        movie_list[0][8]:row[8],
        movie_list[0][9]:row[9],
        movie_list[0][10]:row[10],
        movie_list[0][11]:row[11],
        movie_list[0][12]:row[12],
        movie_list[0][13]:row[13],
        movie_list[0][14]:row[14],
        movie_list[0][15]:row[15],
        movie_list[0][16]:row[16],
        movie_list[0][17]:row[17],
        movie_list[0][18]:row[18],
        movie_list[0][19]:row[19],
        movie_list[0][20]:row[20],
        movie_list[0][21]:row[21]
    }

print("Splitting genre, actors, and writer")
for key in movie_dict.keys():
    temp = []
    for item in movie_dict[key]["genre"].split(","):
        temp.append(item.strip())
    movie_dict[key]["genre"] = temp
    temp = []
    for item in movie_dict[key]["actors"].split(","):
        temp.append(item.strip())
    movie_dict[key]["actors"] = temp
    temp = []
    for item in movie_dict[key]["writer"].split(","):
        temp.append(item.strip())
    movie_dict[key]["writer"] = temp
'''
Not much of a point in continuing to break down categories like above if we don't end up using them.
Pickle file or turn it into a json or something to store it. Get tf-idf for words to use for DNN.
Look into DNN trigram
'''  
print("Dumping file")
dump(movie_dict,open("./data/pickled_data","wb"))
print("Finished Processing raw {}".format(argv[1]))