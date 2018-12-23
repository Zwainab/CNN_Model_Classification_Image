# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 13:32:12 2018

@author: MG
"""
import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing import image
import pyttsx3
from tkinter import Tk, Button, StringVar, Label
# =============================================================================
# 
# =============================================================================
loadModel = load_model( 'classifierModel3.h5' ) 
loadModel.load_weights( 'classferModelWeights3.h5' )
# =============================================================================
# 
# =============================================================================
def funCaptuer():
    capImg = cv2.VideoCapture(0)
    while(True):
        ret, frame = capImg.read()
        redCapImg = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    
        cv2.imshow('frame', redCapImg)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            out = cv2.imwrite('captureImage.jpg', frame)
            break
    return out
    capImg.release()
    cv2.destroyAllWindows()
# =============================================================================
# 
# =============================================================================
def defultPage():
    interfac = Tk()
    interfac.title("Object Recognition for Blind Pepole")
    interfac.geometry("600x600")
    massgPage = StringVar()
    showMassgPage = Label(interfac, textvariable = massgPage, font = 20 )
    massgPage.set("Please blind Pepole chick button Captuer")
    print(massgPage.get())
    showMassgPage.place(x=20,y=20)
    buttonCaptuer = Button(interfac, text = "Captuer", width = 25, command = funCaptuer)
    buttonCaptuer.place(x = 50, y = 50)
    interfac.mainloop()

# =============================================================================
# 
# =============================================================================
def predectedImage():
    loadImg = image.load_img( 'captureImage.jpg' , target_size = ( 64, 64 ))
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
    return objects
        
def showNameObject():
    interfac = Tk()
    interfac.title("Object Recognition for Blind Pepole")
    interfac.geometry("600x600") 
    nameObject = StringVar()
    printObjectName = Label(interfac, textvariable = nameObject, font = 20 )
    nameObject.set(predectedImage())
    print(nameObject.get())
    printObjectName.place(x=100,y=100)
    interfac.mainloop()

def viose():
    read = pyttsx3.init()
    read.setProperty("rate", 60)
    read.say(predectedImage())
    read.runAndWait()

def Exit():
     defultPage()
     
     
def showSelectButton():
    interfac = Tk()
    interfac.title("Object Recognition for Blind Pepole")
    interfac.geometry("600x600") 
    
    nameObject = StringVar()
    printObjectName = Label(interfac, textvariable = nameObject, font = 20 )
    nameObject.set("Want to repeat the sound please chick button Repeat, or  chick button Exit when you want to exit")
    print(nameObject.get())
    printObjectName.place(x=0,y=100)
    
    buttonRepeat = Button(interfac, text = "Repeat", command = viose )
    buttonRepeat.place(x = 100, y = 200)
    
    buttonExit = Button(interfac, text = "Exit", command = Exit )
    buttonExit.place(x = 190, y = 200)
    interfac.mainloop()

def main():
    defultPage()
    predectedImage()
    showNameObject()
    viose()
    showSelectButton()

main()



