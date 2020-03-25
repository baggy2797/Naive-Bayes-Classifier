import numpy as np
import os as os
import string 
from nltk import *
import math as mt

#Specifying the language specific stemmer
stemmer = SnowballStemmer("english")

# Globally defined dictionaries (***Blank Initially***)
_ham_dictionary, _spam_dictionary = {} , {}

# Function to load the dataset
def _load_dataset(file_path):
    words = []
    f_file = open(file_path, 'r')
    for line in f_file:
        for word in line.split():
            if sum(1 for chr in word if chr.islower())>1:
                new = word.strip(string.punctuation)
                new = new.lower()
                words.append(word)
    return words

# Helper Function to list the contents of a directory
def _directory_reading(directory):
    directory_entries = os.listdir(directory)
    return directory_entries

#Function for building the dictionary/vocabulary
def _building_dictionary(path,stopwords,kind,with_stopwords):
    if with_stopwords.lower()=="yes":
        #For Ham Dataset
        if kind.lower() =="ham":
            ham_path = path
            for filename in _directory_reading(ham_path):
                words = _load_dataset(ham_path + filename)
                for word in words:
                    #stemming the word before putting it into the dictionary
                    word = stemmer.stem(word)
                    if word in stopwords:
                        #Checking if it is a stopword
                        continue
                    else:
                        #If word is absent
                        if _ham_dictionary.get(word, 0) == 0:
                            _ham_dictionary[word] = 1
                        #If the word is already present
                        else:
                            _ham_dictionary[word] += 1
        #For Spam Dataset
        if kind.lower() == "spam":
            spam_path = path
            for filename in _directory_reading(spam_path):
                words = _load_dataset(spam_path + filename)
                for word in words:
                    #stemming the word before putting it into the dictionary
                    word = stemmer.stem(word)
                    if word in stopwords:
                        #Checking if it is a stopword
                        continue
                    else:
                        #If word is absent
                        if _spam_dictionary.get(word, 0) == 0:
                            _spam_dictionary[word] = 1
                        #If the word is already present
                        else:
                            _spam_dictionary[word] += 1

    elif with_stopwords.lower()=="no":
        #For Ham Dataset
        if kind.lower() =="ham":
            ham_path = path
            for filename in _directory_reading(ham_path):
                words = _load_dataset(ham_path + filename)
                for word in words:
                    #stemming the word before putting it into the dictionary
                    word = stemmer.stem(word)
                    #If word is absent
                    if _ham_dictionary.get(word, 0) == 0:
                        _ham_dictionary[word] = 1
                    #If the word is already present
                    else:
                        _ham_dictionary[word] += 1
        #For Spam Dataset
        if kind.lower() == "spam":
            spam_path = path
            for filename in _directory_reading(spam_path):
                words = _load_dataset(spam_path + filename)
                for word in words:
                    #stemming the word before putting it into the dictionary
                    word = stemmer.stem(word)
                    #If word is absent
                    if _spam_dictionary.get(word, 0) == 0:
                        _spam_dictionary[word] = 1
                    #If the word is already present
                    else:
                        _spam_dictionary[word] += 1
    else:
        print("\n Enter either 'yes' or 'no'.\n")

# Naive Bayes Classifier Function
def Classify(path):
    classifications = []
    for filename in _directory_reading(path):
        words = _load_dataset(path + filename)
        Prob_ham = 1
        Prob_spam = 1
        for word in words:
            if _ham_dictionary.get(word, 0) + _spam_dictionary.get(word, 0) < 4:
                continue
            else:
                
                p = _ham_dictionary.get(word, 0) / (_ham_dictionary.get(word, 0) + _spam_dictionary.get(word, 0))
                Prob_ham *=p
                q = 1-p
                Prob_spam *= q
        if (Prob_ham) >= (Prob_spam):
            classifications.append(0)
        else:
            classifications.append(1)
    return classifications

def Calc_accuracy(spam_classify,ham_classify):
    spam_len = len(spam_classify)
    ham_len = len(ham_classify)
    total = float(spam_len + ham_len)
    spam_sum, ham_sum = 0,0

    for label in spam_classify:
        if label==1:
            spam_sum = spam_sum + 1

    for label in ham_classify:
        if label==0:
            ham_sum = ham_sum + 1
            
    numerator = float(ham_sum + spam_sum)
    accuracy = 100*float(numerator / total)
    return accuracy

#Loading the Datasets
train_ham_path,train_spam_path = os.sys.argv[1],os.sys.argv[2]
test_ham_path,test_spam_path = os.sys.argv[3],os.sys.argv[4]

#loading the read stopwords
stopwords = []
f = open("stopwords.txt",'r')
for line in f:
    for word in line.split():
        stopwords.append(word)
#print(stopwords)

#Building Dictionaries using stopwords
_building_dictionary(train_ham_path,stopwords,"ham","yes")
_building_dictionary(train_spam_path, stopwords,"spam","yes")

#Classifying as spam or ham
ham_classify,spam_classify = Classify(test_ham_path),Classify(test_spam_path)

#Calculating the Accuracy
print("By Removing stopwords")
acc_1 = Calc_accuracy(spam_classify,ham_classify)
print('Total Accuracy of the dataset is : %f \n' %acc_1)

# Globally re-defined dictionaries,to clear the content (***Blank Initially***)
_ham_dictionary, _spam_dictionary = {} , {}

#Building Dictionaries without using stopwords
_building_dictionary(train_ham_path,stopwords,"ham","no")
_building_dictionary(train_spam_path, stopwords,"spam","no")

#Classifying as spam or ham
ham_classify_wo,spam_classify_wo = Classify(test_ham_path),Classify(test_spam_path)

#Calculating the Accuracy
print("By Keeping stopwords")
acc_2 = Calc_accuracy(spam_classify_wo,ham_classify_wo)
print('Total Accuracy of the dataset is : %f \n' %acc_2)

#writing the results to a text file
sample = open("Accuracy.txt","w")
sample.write("Accuracy (Test_without_stopwords) : " + str(acc_1))
sample.write("\n")
sample.write("Accuracy(Test_with_stopwords) : " + str(acc_2))