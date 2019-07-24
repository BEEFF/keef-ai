from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import logging
logging.getLogger('tensorflow').disabled = True

def print_sep():
    print("--------------------------------------")
    

def fashion():
    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Training Testing Data Shape... #
    print_sep()
    print("What the Train Images look like: ")
    print(train_images.shape)
    print("What that the Train Labels Look Like:")
    print(train_labels)
    print("What the test images look like: ")
    print(test_images.shape)
    print_sep()
    print("Format of data is like a slice. (Samples, Width, Height)")
    print_sep()

    """
    #Pre Processing Data
    plt.figure() #create figure object
    plt.imshow(train_images[1]) #show the first image in the set
    plt.colorbar() #create colour bar
    plt.grid(False) #no grid
    plt.show() # show the figure
    """

    # scale images to range 0 to 1 by dividing by 255
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # display the first 25 images from the training set and dusplay clas sname below the image
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    """
    plt.show()
    """

    # setting up the layers by
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)), #flattening 2d array (24x24) to 1d (784) pixels
        keras.layers.Dense(128, activation=tf.nn.relu), #dense neural layer 128 nodes
        keras.layers.Dense(10, activation=tf.nn.softmax) # 10 node softmax layer, that returns array of 10 probability scores
    ])

    # each of the 10 nodes in the last layers represent a probability of the test image belonging to one of the classes

    # compliing the model
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # model training
    model.fit(train_images, train_labels, epochs=5)
    #a step in training is one gradient update, an epoch is one batch of steps

    # evaluating accuracy
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)
    print_sep()


    # PREDICTIONS
    predictions = model.predict(test_images)
    #Here, the model has predicted the label for each image in the testing set. Let's take a look at the first prediction:
    print(predictions[0])
    print_sep()
    print("Lets predict... ")
    predicted_label = (np.argmax(predictions[0])) #get the maximum probability of the image belonging to a specific class
    print(class_names[predicted_label])

def organise_playing_cards():













