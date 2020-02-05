# Adversary Attack

# importing the libraries

import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.inception_v3 import InceptionV3, decode_predictions
from keras import backend as K
from keras.preprocessing import image

# We load the model, if this is the first time you use it, this can be slower because it has to be downloaded

iv3 = InceptionV3()

# load and convert the image into a matrix of numerical values (Beer)
# target_size=(299, 299) is for do a reshape to make it compatible with the model InceptionV3

X = image.img_to_array(image.load_img("C:/Users/chris/PycharmProjects/MachineLearningA-ZCourse/"
                                      "IA Notebook/IA Notebook 2/beer.jpg", target_size=(299, 299)))

# range change, 0 - 255 -> -1 - 1
X /= 255
X -= 0.5
X *= 2

"""The method to convert the image on a tensor have to take 4 arguments the first one is
for if you want to pass more than one image at a time"""

X.reshape([1, X.shape[0], X.shape[1],  X.shape[2]])

y = iv3.predict(X)

decode_predictions(y)

# load and convert the image into a matrix of numerical values (Cat)
# target_size=(299, 299) is for do a reshape to make it compatible with the model InceptionV3

X = image.img_to_array(image.load_img("C:/Users/chris/PycharmProjects/MachineLearningA-ZCourse/"
                                      "IA Notebook/IA Notebook 2/cat.jpg", target_size=(299, 299)))

# range change, 0 - 255 -> -1 - 1
X /= 255
X -= 0.5
X *= 2

"""The method to convert the image on a tensor have to take 4 arguments the first one is
for if you want to pass more than one image at a time"""

X.reshape([1, X.shape[0], X.shape[1],  X.shape[2]])

y = iv3.predict(X)

decode_predictions(y)

# Starting with the Adversary Attack

# We take the input image and the output
input_layer = iv3.layers[0].input
output_layer = iv3.layers[-1].output

# Define the target class that in this case will be a lemon
target_class = 951

# In this line we are telling that the function coast that we want to increase is the class 951

loss = output_layer[0, target_class]
grad = K.gradients(loss, input_layer)[0]

# The image that we want to optimize
optimize_gradient = K.function([input_layer, K.learning_phase()], [grad, loss])

adv = np.copy(X)

"""With this we are telling to our optimizer that the image have to be in a range defined by
max_pert - min_per. With this the human eye can't see the deference"""

pert = 0.01

max_pert = X + 0.01
min_pert = X - 0.01

cost = 0.0

"""We will have at less a probability of 95 percentage that the CNN preaches that the image is a lemon when 
the image contains a cat"""
while cost < 0.95:
    gr, cost = optimize_gradient([adv, 0])
    adv += gr
    adv = np.clip(adv, max_pert, min_pert)
    adv = np.clip(adv, -1, 1)
    print("Lemon cost: ", cost)
hacked = np.copy(adv)

# range change, -1 - 1 -> 0 - 255

adv /= 2
adv += 0.5
adv *= 255

# Visualising the hacked image

plt.imshow(adv[0].astype(np.uint8))
plt.show()

# Predicting the hacked image
# load and convert the image into a matrix of numerical values (Hacked)
# target_size=(299, 299) is for do a reshape to make it compatible with the model InceptionV3

X = image.img_to_array(image.load_img("C:/Users/chris/PycharmProjects/MachineLearningA-ZCourse/"
                                      "IA Notebook/IA Notebook 2/cat.jpg", target_size=(299, 299)))

# range change, 0 - 255 -> -1 - 1
X /= 255
X -= 0.5
X *= 2

"""The method to convert the image on a tensor have to take 4 arguments the first one is
for if you want to pass more than one image at a time"""

X.reshape([1, X.shape[0], X.shape[1],  X.shape[2]])

y = iv3.predict(X)

decode_predictions(y)
