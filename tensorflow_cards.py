## Developed by Thomas O'Keeffe #
## Modules ##
from google_images_download import google_images_download
import tensorflow as tf
#from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import logging
from PIL import Image

def get_card_names():
    suits = ["Hearts,", "Clubs", "Diamonds", "Spades"]
    numbers = ["Ace", "King", "Queen", "Jack", "Ten", "Nine", "Eight", "Seven", "Six", "Five", "Four", "Three", "Two"]
    terms = []
    for number in numbers:
        for suit in suits:
            terms.append(number + " of " + suit + " card")
    return terms

def get_google_images(search_term, number=5):
    response = google_images_download.googleimagesdownload()   #class instantiation
    arguments = {"keywords":search_term,"limit":number,"print_urls":False}   #creating list of arguments
    paths = response.download(arguments)   #passing the arguments to the function

def scrape_cards_data():
    card_names = get_card_names()
    for card in card_names:
        get_google_images(card, 15)

def image_resize(path):
    basewidth = 200 # MNIST image width
    img = Image.open(path)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    return img

def training():
    class_labels = get_card_names()

image_resize()
