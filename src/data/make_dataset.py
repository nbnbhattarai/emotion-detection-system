'''
Download dataset
'''
import os
import gdown

import pandas as pd
import numpy as np
from data.dataset import Dataset

DATASET_URL = 'https://drive.google.com/uc?id=1l_J0P9A_AD8d_rzZHJ5Fg8F4y1nGP_x3'
DATASET_FILENAME = '../data/raw/emotion_raw.txt'


def download_dataset(filename, rewrite=False):
    '''
    Download dataset
    '''

    if not rewrite and os.path.exists(filename):
        # log.info(f'Dataset file {filename} already exists.')
        return True

    try:
        gdown.download(DATASET_URL, filename, quiet=True)
        # logger.success(f'Dataset file {filename} downloaded successfully.')
    except Exception as download_exception:
        # logger.error(f'Exception occured: {download_exception}')
        print(f'Exception {download_exception} occured!')
        return False

    return True


def load_emotion():
    '''
    Load the emotion dataset and return bunch object.
    '''

    download_success = download_dataset(DATASET_FILENAME)

    if download_success:
        columns = ['emotion', 'text']
        emo_data = pd.read_csv(DATASET_FILENAME, names=columns)

        x_raw = emo_data['text'].values
        y_raw = emo_data['emotion'].values

        return Dataset(x=x_raw, y=y_raw)

    return None
