'''
Open the glove feature file if it exists otherwise download glove vector and return
'''
import os
import logging

import gdown

GLOVE_FILENAME = '../../data/raw/glove_word2vec.txt'
GLOVE_FILEURL = 'https://drive.google.com/uc?id=1E2FCguEoggAVak1dCXksXlfq7ruOiFU1'

logger = logging.getLogger(__name__)


def download_glove_features():
    '''
    Download the glove word2vec file
    '''
    try:
        gdown.download(GLOVE_FILEURL, GLOVE_FILENAME, quiet=True)
    except Exception as e:
        logger.error(
            f'Exception occured while downloading {GLOVE_FILENAME} from {GLOVE_FILEURL} : {e}')
        return False
    return True


def get_glove():
    '''
    Return the glove word encoding
    '''
    word2vec = {}
    file_status = True
    if not os.path.exists(GLOVE_FILENAME):
        file_status = download_glove_features()

    if file_status:
        with open(GLOVE_FILENAME) as fl:
            for l in fr:
                line = l.split()
                word = line[0]
                word_vec = np.array(line[1:], dtype=np.float64)
                word2vec[word] = word_vec

        return word2vec
    return None
