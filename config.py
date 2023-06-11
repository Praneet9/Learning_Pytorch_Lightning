import os

# Training Hyperparameters
DEV_MODE = True
N_CLASSES = 12
LR = 0.001
BATCH_SIZE = 64
N_EPOCHS = 10

# Dataset Config
DATA_DIR = os.path.join('data', 'speechImageData')
TRAIN_DIR = os.path.join(DATA_DIR, 'TrainData')
TEST_DIR = os.path.join(DATA_DIR, 'ValData')
HEIGHT = 98
WIDTH = 50
NUM_WORKERS = 8

# Computation
ACCELERATOR = 'gpu'
DEVICES = [0]
PRECISION = 32