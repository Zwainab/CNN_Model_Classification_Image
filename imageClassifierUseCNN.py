# -*- coding: utf-8 -*-


# =============================================================================
# Step 1 Import Libraries.
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# =============================================================================
# Step 2 Load Datasete.
# =============================================================================
folderTrining = ('training_images/training')
folderTesting = ('training_images/testing')


# =============================================================================
# Step 3 Build and Train Models using Design CNN Architecture
# =============================================================================
classifierModel=Sequential()
#steps CNN:
#step 1 use Convolution operation-> for extract features(is filtering)
classifierModel.add(Convolution2D(32,(3,3), input_shape = (64, 64, 3)))
classifierModel.add(Activation('relu'))
classifierModel.add(MaxPooling2D(pool_size = (2, 2)))# Step 2 - Pooling

# Adding a second convolutional layer
classifierModel.add(Convolution2D(32, (3, 3)))
classifierModel.add(Activation('relu'))
classifierModel.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifierModel.add(Flatten())#14x14x32 = 6272. convert 2D matrix to 1D vector.
classifierModel.add(Dropout(0.25))

# Step 4 - Full connection is hidden layer
classifierModel.add(Dense(128))
classifierModel.add(Activation('relu'))
classifierModel.add(Dropout(0.5))
classifierModel.add(Dense(1))
classifierModel.add(Activation('sigmoid'))#output layer use one node

# =============================================================================
# Compiling the CNN Evaluate Model Accuracy
# =============================================================================

classifierModel.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])





# =============================================================================
# Part 2 - Fitting the CNN to the images
# =============================================================================
train_data = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_data = ImageDataGenerator(rescale = 1./255)

training_set = train_data.flow_from_directory(folderTrining,
                                                 target_size = (64, 64),
                                                 batch_size = 20,
                                                 class_mode = 'binary')

test_set = test_data.flow_from_directory(folderTesting,
                                            target_size = (64, 64),
                                            batch_size = 20,
                                            class_mode = 'binary')

# =============================================================================
# Train the Model
# here traning: for training Keras a model using Python data generators.
# =============================================================================

trninModel = classifierModel.fit_generator(training_set,
                         samples_per_epoch = 100,
                         nb_epoch = 20,
                         validation_data = test_set,
                         nb_val_samples = 50,
                         verbose= 1)



# =============================================================================
# Save Model 
# =============================================================================


saveModel = classifierModel.save('classifierModel3.h5') 
saveModelWeights = classifierModel.save_weights('classferModelWeights3.h5')



# =============================================================================
# Display summary accuracy using history.
# =============================================================================
print(trninModel.history.keys())

plt.plot( trninModel.history['acc'] )
plt.plot( trninModel.history['val_acc'] )
plt.title( 'Model Accuracy' )
plt.ylabel( 'Accuracy' )
plt.xlabel( 'epoch' )
plt.legend( [ 'train','test' ], loc = 'upper right' )
plt.show()
# =============================================================================
# Display summary loss using history.
# =============================================================================

plt.plot( trninModel.history['loss'] )
plt.plot( trninModel.history['val_loss'] )
plt.title( 'Model Loss' )
plt.ylabel( 'Loss' )
plt.xlabel( 'epoch' )
plt.legend( [ 'train','test' ], loc = 'best' )
plt.show()















