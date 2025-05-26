import os

HOME_PATH   =   os.path.expanduser("~")
JOB_ID      =   "4050laptop-unet"
MODEL_NAME  =   "cityscapeseg"
DS_PATH     =   "/home/cbalfour/Group5_p2/data"
OUT_PATH    =   "/home/cbalfour/Group5_p2/outputs"

JOB_FOLDER = os.path.join(OUT_PATH, JOB_ID)
TRAINED_MDL_PATH = os.path.join(JOB_FOLDER, "parameters")
BATCH_SIZE = 16
LR = 1e-4
LOG_BATCH_INTERVAL = 1
LOG_WANDB = True
NUM_WORKERS = 1
