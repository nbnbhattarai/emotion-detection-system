import argparse
import logging

import torch

import mlflow

from settings import get_logger

import pickle

parser = argparse.ArgumentParser()

parser.add_argument(
    "model", help="select model from (naive, svm, logistic)")

parser.add_argument('-l', '--log-level', type=str,
                    help='Select log level from (debug, info, warning, error, critical)', default='error')

parser.add_argument(
    "-t", "--test-size", type=float, help="Proportion of dataset to use as test set"
)
parser.add_argument('-k', '--kernel', type=str, help='Kernel name for svm')
parser.add_argument('-d', '--degree', type=int,
                    help='Degree of polynomial kernel')
parser.add_argument('-c', '--cost', type=float, help='Cost parameter for svm')
parser.add_argument('-g', '--gamma', type=float,
                    help='Kernel coefficient for rbf, poly and sigmoid kernel')
parser.add_argument('-s', '--save-model', type=str,
                    help='Save model to filename')
parser.add_argument('--data-augmentation', type=str,
                    help='Data augmentation type from wordnet, bert, ...')
parser.add_argument('--augment-size', type=float,
                    help='Propertion of data to augment. Default to 0.2 of one-fifth.')

args = parser.parse_args()

LOG_LEVELS_MAP = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL,
}

LOGGER = get_logger('', LOG_LEVELS_MAP.get(args.log_level))

LOGGER.debug(f'Model argument: {args.model}')

MODEL_FILENAME = args.save_model
DATA_AUGMENTATION = args.data_augmentation
AUGMENTATION_SIZE = 0

if DATA_AUGMENTATION:
    AUGMENTATION_SIZE = args.augment_size or 0.2

if args.model == 'naive':
    LOGGER.info('Using naive bayes model')

    exp_name = 'Naive Bayes'

    mlflow.set_experiment(exp_name)

    test_size = None

    if args.test_size:
        test_size = args.test_size
        LOGGER.info(f'Test size: {test_size}')
        mlflow.log_param(f'Test size', test_size)

    if DATA_AUGMENTATION:
        mlflow.log_param(f'AUGMENTATION_SIZE', AUGMENTATION_SIZE)
        LOGGER.info(f'Augmentation size: {AUGMENTATION_SIZE}')

    from models.naive_bayes import NaiveBayesEmotionDetection

    with mlflow.start_run() as run:
        naive_model = NaiveBayesEmotionDetection(
            test_size, DATA_AUGMENTATION, AUGMENTATION_SIZE)
        LOGGER.info(f'Experiment {exp_name} started.')
        naive_model.experiment()

        if MODEL_FILENAME:
            with open(MODEL_FILENAME) as fl:
                pickle.dump(naive_model.model, fl)
            LOGGER.info(
                f'Model {exp_name} saved to {MODEL_FILENAME} successfully!')

        LOGGER.info(f'Experiment {exp_name} completed.')

if args.model == 'svm':
    LOGGER.info(f'Using SVM model')

    exp_name = 'SVM'

    mlflow.set_experiment(exp_name)

    C = args.cost
    kernel = args.kernel
    degree = args.degree
    gamma = args.gamma

    LOGGER.info(
        f'Set parameters to: C={C}, kernel={kernel}, degree={degree}, gamma={gamma}')

    test_size = None
    if args.test_size:
        test_size = args.test_size
        LOGGER.info(f'Test size: {test_size}')
        mlflow.log_param(f'Test size', test_size)

    mlflow.log_param('C', C)
    mlflow.log_param('kernel', kernel)
    mlflow.log_param('degree', degree)
    mlflow.log_param('gamma', gamma)

    from models.svm import SVMEmotionDetection
    # with mlflow.start_run() as run:
    svm_model = SVMEmotionDetection(
        test_size=test_size, C=C, kernel=kernel, degree=degree, gamma=gamma)
    LOGGER.info(f'Experiment {exp_name} started.')
    svm_model.experiment()
    LOGGER.info(f'Experiment {exp_name} completed.')


if args.model == 'rnn':
    from models.rnn import RNNModel
    model = RNNModel(args.test_size)
    model.experiment(n_epochs=300)
