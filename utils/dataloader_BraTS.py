# todo: add relevant libs
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

SEGMENT_CLASS = {
   0: 'NOT TUMOR',
   1: 'NECROTIC/CORE',
   2: 'EDEMA',
   4: 'ENHANCING'
}

VOLUME_SLICES = 155
VOLUME_START_AT = 22

TRAIN_PATH = "../data/train/"
