'''Module for model inference'''
import pickle
import sys
import logging

from data.make_dataset import load_emotion


LOGGER = logging.getLogger(__name__)


def model_inference(text, model_name):
    '''
    Return probability for individual emotion category
    '''
    model_name = model_name.lower()
    if model_name in ['svm']:
        pass
    elif model_name in ['naive', 'naive_bayes']:
        pass
    elif model_name in ['bert']:
        pass
    elif model_name in ['rnn']:
        pass


def naive_inference(in_text):
    '''Inference for text using SVM'''
    from sklearn.naive_bayes import MultinomialNB
    from features.build_features import process_text
    from sklearn.feature_extraction.text import CountVectorizer

    # cvector = CountVectorizer(tokenizer=process_text)

    # cvector.fit(dataset.data)

    with open('artifacts/cvector.pkl', 'rb') as fl:
        cvector = pickle.load(fl)

    LOGGER.info(f'type: {type(cvector)}')
    x_vec = cvector.transform([in_text])

    LOGGER.info(f'x_vec: {x_vec.toarray()}')

    mpkl_f = open('artifacts/naive.pkl', 'rb')
    model = pickle.load(mpkl_f)

    output = model.predict(x_vec)

    predict = model.labels[output[0]]
    print(f'prediction: {predict}')

    return in_text, predict


if __name__ == '__main__':
    text = sys.argv[1]

    naive_predict = naive_inference(text)

    print(f'Naive bayes prediction: {naive_predict}')
