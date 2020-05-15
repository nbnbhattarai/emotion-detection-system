'''
Build features from the dataset
'''

import os
import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from autocorrect import Speller

from sklearn.feature_extraction.text import CountVectorizer

nltk.download('punkt')
nltk.download('stopwords')

DATASET_FILE = 'data/'

en_speller = Speller(lang='en')

EN_STOPWORDS = stopwords.words('english')


def process_text(text):
    '''
    Process text and return tokenized list of words
    '''
    text = text if isinstance(text, str) else ''

    # clean the words, remove symbols special chars
    pattern = r'[^a-zA-z0-9\s]'
    text = re.sub(pattern, '', text)

    # convert to lowercase
    text = text.lower()

    # first tokenize the text
    word_tokenized = word_tokenize(text)

    # let's remove the stop words
    # print(f'Word tokenized: {word_tokenized[:20]}')
    words_swords_removed = [
        en_speller(word) for word in word_tokenized if word not in EN_STOPWORDS]

    return words_swords_removed


def get_features(dataset):
    '''
    Convert given dataset into count vector matrix
    '''
    x_raw = dataset.data
    y_target = dataset.target

    cvector = CountVectorizer(tokenizer=process_text)

    x_vec = cvector.fit_transform(x_raw)

    return x_vec.toarray(), y_target