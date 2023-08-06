# By Juichi Lee

import string
import re
import numpy as np
import math
import os

# global variables
# vocabulary = None
# vocabSize = None

def main():
  
  # text preprocessing step
  vocabulary, vocabSize = PreprocessTrainingText('in/trainingSet.txt')
  
  # training and testing
  trainingData = GetTextFileText('in/trainingSet.txt')
  testData = GetTextFileText('in/testSet.txt')
  trainingList = turnIntoFeatureList(trainingData, vocabulary, vocabSize)
  testList = turnIntoFeatureList(testData, vocabulary, vocabSize)

  outputVectorList('./out/preprocessed_train.csv', vocabulary, trainingList)
  outputVectorList('./out/preprocessed_test.csv', vocabulary, testList)
  
  pClassLabelTrue, pClassLabelFalse, tableOfPTrueGivenCLTrue, tableOfPTrueGivenCLFalse = NaiveBayes(trainingList, vocabSize)

  trainingAccuracy = GetAccuracy(trainingList, pClassLabelTrue, pClassLabelFalse, tableOfPTrueGivenCLTrue, tableOfPTrueGivenCLFalse, vocabSize)
  testAccuracy = GetAccuracy(testList, pClassLabelTrue, pClassLabelFalse, tableOfPTrueGivenCLTrue, tableOfPTrueGivenCLFalse, vocabSize)

  # outputting results
  if not os.path.exists("./out"):
    os.makedirs("./out")

  with open("./out/results.txt", 'w') as file:
    file.write("Accuracy from training on trainingSet.txt and testing on trainingSet.txt: " + str(trainingAccuracy) + "\n")
    file.write("Accuracy from training on trainingSet.txt and testing on testSet.txt: " + str(testAccuracy) + "\n")
    file.close()
  
  # user input to test trained Naive Bayes Model
  loop = True
  
  while (loop):
    inputReview = input("Please enter a review to evaluate sentiment: ")
    inputData = turnIntoFeatureVector(inputReview, vocabulary)
    sentiment = Classification(inputData, pClassLabelTrue, pClassLabelFalse, tableOfPTrueGivenCLTrue, tableOfPTrueGivenCLFalse, vocabSize)
    print(f"Sentiment(0=neg/1=pos): {sentiment}\n")
    
    while True:
      continueLoop = input("Would you like to input another review? (y/n): ")
      if continueLoop.lower() == "y":
        break
      if continueLoop.lower() == "n":
        loop = False
        break
      print("Please input y or n")

def PreprocessTrainingText(filename):
  with open(filename, 'r') as file:
    data = file.read().replace('\n', '')
    file.close

  exclude = set(string.punctuation) 
  data = ''.join(ch for ch in data if ch not in exclude) # remove all punctuation

  data = data.lower() # make all chars lowercase

  processedData = ''.join([i for i in data if not i.isdigit()]) # remove all numbers

  listData = re.sub("[^\w]", " ",  processedData).split() # remove all non-alphanumeric chars and create list of words
  listData = list(set(listData)) # make list of words a set
  listData.sort() 
  
  # global vocabulary
  vocabulary = listData
  
  # global vocabSize
  vocabSize = len(vocabulary)
  print("VocabSize: ", vocabSize)
  return vocabulary, vocabSize


# Summary: Based on NaiveBayes approach to ML and uses Bayes Rule to determine probability weights
# Input: trainingList (List(List(int))), testList (List(List(int))), vocabSize (int)
# Output: (trainingAccuracy (float), testAccuracy (float))
def NaiveBayes(trainingList, vocabSize):
  # Training
  trainingListSize = np.size(trainingList, 0)
  numCLTrue = 0
  for line in trainingList: 
    if(line[-1] == "1"):
      numCLTrue = numCLTrue + 1

  # calculate True and False classification probabilities
  pClassLabelTrue = float(numCLTrue) / trainingListSize 
  pClassLabelFalse = float(1.0) - pClassLabelTrue

  # stores the probabilities of words classifying as True or False
  tableOfPTrueGivenCLTrue = []
  tableOfPTrueGivenCLFalse = []

  # go thru each word in the vocab 
  for vocabidx in range(vocabSize):
    bothTrueCounter = 0
    mixedCounter = 0
    
    for line in trainingList:
      if(line[-1] == "1" and line[vocabidx] == "1.0"):
        bothTrueCounter = bothTrueCounter + 1

      if(line[-1] == "0" and line[vocabidx] == "1.0"):
        mixedCounter = mixedCounter + 1
 
    pWordTrueGivenCLTrue = float(bothTrueCounter + 1)/(numCLTrue + 2)
    pWordTrueGivenCLFalse = float(mixedCounter + 1)/(trainingListSize - numCLTrue + 2)
      
    tableOfPTrueGivenCLTrue.append(pWordTrueGivenCLTrue)
    tableOfPTrueGivenCLFalse.append(pWordTrueGivenCLFalse)

  return pClassLabelTrue, pClassLabelFalse, tableOfPTrueGivenCLTrue, tableOfPTrueGivenCLFalse

def GetAccuracy(testList, pClassLabelTrue, pClassLabelFalse, tableOfPTrueGivenCLTrue, tableOfPTrueGivenCLFalse, vocabSize):
  numCorrect = 0
  for line in testList:
    predicted = Classification(line, pClassLabelTrue, pClassLabelFalse, tableOfPTrueGivenCLTrue, tableOfPTrueGivenCLFalse, vocabSize)

    # check if prediction was correct
    if(predicted == line[-1]):
      numCorrect = numCorrect + 1

  accuracy = float(numCorrect)/np.size(testList, 0) # calculate overall accuracy
  print("Accuracy: ", accuracy)
  return accuracy

# Summary: Attempts to classify new sentences using previously trained parameters, compares predictions to testList and returns an overall accuracy
# Input: testList (List(List(int))), pClassLabelTrue (float), pClassLabelfalse (float), tableofPTrueGivenCLTrue (List(float)), tableOfPTrueGivenCLFalse (List(float))
# Output: accuracy (float)
def Classification(line, pClassLabelTrue, pClassLabelFalse, tableOfPTrueGivenCLTrue, tableOfPTrueGivenCLFalse, vocabSize):
  # Classification
  # general formula: PCLTrue * PWord1TrueGivenCLTrue * PWord2TrueGivenClTrue etc...
  # apply log: log(PCLTrue) + log(PWord1TrueGivenCLTrue) + log(PWord2TrueGivenClTrue) etc...
  wordPresent = []
  for wordIdx in range(vocabSize):
    if(line[wordIdx] == "1.0"):
      wordPresent.append(wordIdx)
      
  # Calculating probability CL = 1
  pCL1 = math.log(pClassLabelTrue)
  for wordIdx in wordPresent:
    pCL1  = pCL1 + math.log(tableOfPTrueGivenCLTrue[wordIdx])
    
  # Calculating probability CL = 0
  pCL0 = math.log(pClassLabelFalse)
  for wordIdx in wordPresent:
    pCL0  = pCL0 + math.log(tableOfPTrueGivenCLFalse[wordIdx])

  # generate prediction
  predicted = None 
  if (pCL1 >= pCL0): # if probability CL1 >= probability CL0, prediction is 1
    predicted = "1"
  else:
    predicted = "0"
  return predicted

def GetTextFileText(fileName):
  with open(fileName, 'r') as file:
    data = file.readlines()
    file.close()
  return data

def turnIntoFeatureList(data, vocabulary, vocabSize):
  vectorList = np.zeros((0, vocabSize+1)) # position @ vocabSize + 1 used to store predictions about feature vector
  
  for i in range(len(data)):
    processedLine = data[i].lower()
    vectorData = turnIntoFeatureVector(processedLine, vocabulary)
    vectorList = np.append(vectorList, np.array([vectorData]), 0)
  print("VectorList Shape: ", vectorList.shape)
  
  return vectorList
    
# Summary: Preprocesses a valid training/testing input file of sentences, turns them into feature vectors, and stores them into feature vector list
# Input: fileName (string)
# Output: vectorList (List(List(int)))
def turnIntoFeatureVector(processedLine, vocabulary):
  
  processedLineList = processedLine.split()
  classLabel = processedLineList[-1]
  
  # preprocess current sentence into a word array
  exclude = set(string.punctuation + string.digits)
  processedLine = ''.join(ch for ch in processedLine if ch not in exclude)
  wordArray = processedLine.split()
  
  # create a list of 0s and 1s on the fly
  # "1" indicate the word is present in the vocabulary, otherwise "0"
  vectorData = np.zeros((0,0))
  for word in vocabulary:
    if word in wordArray:
      vectorData = np.append(vectorData, 1)
    else:
      vectorData = np.append(vectorData, 0)

  # append classification 
  vectorData = np.append(vectorData, classLabel) 
  
  return vectorData


# Summary: Outputs the results from a vectorList to a file specified in filepath
# Input: filepath (string), vocab (List(string)), vectorList (List(List(int)))
# Output: None (writes to an output file)
def outputVectorList(filepath, vocab, vectorList):
  print("OutputVectorList: ", filepath)
  vocabString = ','.join(vocab)
  vocabString += ', ClassLabel(0=neg/1=pos)'
  with open(filepath, 'w') as file:
    file.write(vocabString + "\n")

    for line in vectorList:
      vectorLine = line.tolist()
      prettyLine = ",". join(vectorLine)
      file.write(prettyLine + "\n")
    file.close()

if __name__ == "__main__":
  main()

# How to strip punctuation from a string
# s.translate(None, string.punctuation)

# How to make string lowercase
# string.lower()

# How to remove numbers from a string
# string = 'abcd1234efg567'
# newstring = ''.join([i for i in string if not i.isdigit()])

# How to make a list of words from a string
# import re
# mystr = 'This is a string, with words!'
# wordList = re.sub("[^\w]", " ",  mystr).split()

# How to sort a list of strings in alphabetical order
# s.sort()