import argparse
import mlflow

parser = argparse.ArgumentParser()

parser.add_argument(
    "model", help="select model from (naive, svm, logistic)")

parser.add_argument(
    "-t", "--test-size", type=float, help="Proportion of dataset to use as test set"
)

args = parser.parse_args()

test_size = None

if args.test_size:
    test_size = args.test_size
    mlflow.log_param(f'Test size', test_size)

if args.model == 'naive':
    from models.naive_bayes import NaiveBayesEmotionDetection
    naive_model = NaiveBayesEmotionDetection(test_size)
    naive_model.experiment()
