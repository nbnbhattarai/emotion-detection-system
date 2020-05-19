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

args = parser.parse_args()

log_levels_map = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL,
}

logger = get_logger('', log_levels_map.get(args.log_level))

logger.debug(f'Model argument: {args.model}')

if args.model == 'naive':
    logger.info('Using naive bayes model')

    exp_name = 'Naive Bayes'

    mlflow.set_experiment(exp_name)

    test_size = None

    if args.test_size:
        test_size = args.test_size
        logger.info(f'Test size: {test_size}')
        mlflow.log_param(f'Test size', test_size)

    from models.naive_bayes import NaiveBayesEmotionDetection

    with mlflow.start_run() as run:
        naive_model = NaiveBayesEmotionDetection(test_size)
        logger.info(f'Experiment {exp_name} started.')
        naive_model.experiment()
        with open('./artifacts/naive.pkl', 'wb') as fl:
            pickle.dump(naive_model.model, fl)

        with open('./artifacts/cvector.pkl', 'wb') as fl:
            pickle.dump(naive_model.cvector, fl)

        logger.info(f'Experiment {exp_name} completed.')

if args.model == 'svm':
    logger.info(f'Using SVM model')

    exp_name = 'SVM'

    mlflow.set_experiment(exp_name)

    C = args.cost
    kernel = args.kernel
    degree = args.degree
    gamma = args.gamma

    logger.info(
        f'Set parameters to: C={C}, kernel={kernel}, degree={degree}, gamma={gamma}')

    test_size = None
    if args.test_size:
        test_size = args.test_size
        logger.info(f'Test size: {test_size}')
        mlflow.log_param(f'Test size', test_size)

    mlflow.log_param('C', C)
    mlflow.log_param('kernel', kernel)
    mlflow.log_param('degree', degree)
    mlflow.log_param('gamma', gamma)

    from models.svm import SVMEmotionDetection
    # with mlflow.start_run() as run:
    svm_model = SVMEmotionDetection(
        test_size=test_size, C=C, kernel=kernel, degree=degree, gamma=gamma)
    logger.info(f'Experiment {exp_name} started.')
    svm_model.experiment()
    logger.info(f'Experiment {exp_name} completed.')


if args.model == 'rnn':
    from models.rnn import RNNModel
    model = RNNModel(args.test_size)
    model.experiment(n_epochs=300)
