import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import argparse
import json
from imutils import paths

dir = "/home/alexandre/Documentos/TESE/results/2nd_Try/train_losses/"

mask_loss_path = dir + "mrcnn_mask_loss.json"
val_mask_loss_path = dir + "val_mrcnn_mask_loss.json"

with open(mask_loss_path) as json_file:
    data_mask_loss = json.load(json_file)
with open(val_mask_loss_path) as json_file:
    data_val_mask_loss = json.load(json_file)

mask_loss = []
val_mask_loss = []

for k in range(len(data_mask_loss)):
    mask_loss.append(data_mask_loss[k][2])
    val_mask_loss.append(data_val_mask_loss[k][2])

plt.plot(mask_loss)
plt.plot(val_mask_loss)
plt.legend(('mask_loss', 'val_mask_loss'), loc='upper right')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()