import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import keras
import pandas as pd
import sklearn
from keras.layers import Flatten, Dense
from keras import layers
from sklearn.metrics import confusion_matrix
from google.colab import drive
import os,shutil,pathlib
from keras.utils import image_dataset_from_directory
import random
class_list=train_dataset = image_dataset_from_directory("catdog")
print(class_list.class_names)