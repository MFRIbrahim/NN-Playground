import torch

LEARNING_RATE = 0.1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAIN_BATCH_SIZE = 8
VAL_BATCH_SIZE = 16
N_EPOCHS = 100
N_WORKERS = 8
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = 'data/train_img/'
VAL_IMG_DIR = 'data/val_img/'
TEST_IMG_DIR = 'data/test_img/'
MODEL_DIR = 'models/'

CHANNEL_MEANS = [0.4700, 0.4468, 0.4076]
CHANNEL_STDS = [0.2750, 0.2704, 0.2854]