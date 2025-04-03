# config.py
import os

# Data paths
BASE_DIR = "C:/brain_tumor_project/data/pickle"
TRAIN_PICKLE = os.path.join(BASE_DIR, "training_data.pickle")

# Hyperparameters
BATCH_SIZE = 32  # training data is divided into mini-batches of 32 samples
EPOCHS = 10      # number of times the model goes through the dataset
LEARNING_RATE = 1e-3  # how much the model's weights are adjusted during each update
NUM_CLASSES = 3       # indicates that the model is designed to classify into 3 different categories

# Image size
IMG_SIZE = (224, 224)
