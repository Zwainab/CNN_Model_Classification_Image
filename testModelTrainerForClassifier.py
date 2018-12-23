# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 19:49:13 2018

@author: MG
"""
from keras.models import load_model
from keras.preprocessing import image
import os
import numpy as np
import pyttsx3


# =============================================================================
# Load Model
# =============================================================================



loadModel = load_model( 'classifierModel3.h5' ) 
loadModel.load_weights( 'classferModelWeights3.h5' )
print( loadModel.summary() )



# =============================================================================
# Testing 
# =============================================================================

def Testing(file):
    testImg = image.load_img(file, target_size=(64,64))
    testImg = image.img_to_array(testImg)
    testImg = np.expand_dims(testImg, axis=0)
    array = loadModel.predict(testImg)#call fun predict.
    result = array[0]
    
    if result[0] == 0:
        print("Predicted answer: cucumber")
        answer = 'cucumber'
    elif result[0] == 1:
        print("Predicted answer: eggplant")
        answer = 'eggplant'
        
    return answer


Tp = 0
Tn = 0
Fp = 0
Fn = 0

for num, name in enumerate(os.walk('training_images/testing/cucumber')):
  for num, filename in enumerate(name[2]):
    if filename.startswith("."):
      continue
    print("Label: cucumber", num)
    result = Testing(name[0] + '/' + filename)
    if result == "cucumber":
      Tn += 1
    else:
      Fp += 1

for num, name in enumerate(os.walk('training_images/testing/eggplant')):
  for num, filename in enumerate(name[2]):
    if filename.startswith("."):
      continue
    print("Label: eggplant", num)
    result = Testing(name[0] + '/' + filename)
    if result == "eggplant":
      Tp += 1
    else:
      Fn += 1
# Performance using Precision-Recall 
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
# Precision is: a measure of result relevancy.
# Recall is:  is a measure of how many truly relevant results are returned.
print("True Positive: ", Tp)
print("True Negative: ", Tn)
print("False Positive: ", Fp)  
print("False Negative: ", Fn)

Precision = Tp / (Tp + Fp)
Recall = Tp / (Tp + Fn)
print("The Precision = ", Precision)
print("The Recall = ", Recall)

measure = (2 * Recall * Precision) / (Recall + Precision)
print("The Measure = ", measure)
# =============================================================================
# New Predict
# =============================================================================
loadImg = image.load_img( 'eggplantNew.jpg', target_size = ( 64, 64 ))
imagePredct = image.img_to_array( loadImg )
imagePredct = np.expand_dims( imagePredct, axis = 0 )
array = loadModel.predict(imagePredct)
result = array[0]
#training_set.class_indices how label

if result[0] == 0:
    print("cucumber")
    objects = "cucumber"
else:
    print("eggplant")
    objects = "eggplant"

read = pyttsx3.init()
read.setProperty("rate", 60)
read.say(objects)
read.runAndWait()
select = int(input("Want to repeat the sound please enter 1, or 0 when you want to exit"))
if select == 1:
    read.say(objects)
    read.runAndWait()
else:
     exit(0)
    

