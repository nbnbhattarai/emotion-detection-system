from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from data.make_dataset import load_emotion
from features.build_features import get_features

from utils.logreport import log_classification_report

import mlflow


class NaiveBayesEmotionDetection():
    def __init__(self, test_size=None):
        self.test_size = test_size or 0.25

    def experiment(self):
        self.model = MultinomialNB()
        self.dataset = load_emotion()
        self.X, self.y = get_features(self.dataset)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size)

        self.model.fit(self.X_train, self.y_train)

        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)

        id2label = dict(enumerate(self.dataset.labels))

        ytrain = [id2label[y] for y in self.y_train]
        ytrain_pred = [id2label[y] for y in y_train_pred]

        ytest = [id2label[y] for y in self.y_test]
        ytest_pred = [id2label[y] for y in y_test_pred]

        report = {}
        report['train'] = classification_report(
            ytrain, ytrain_pred, output_dict=True)
        report['test'] = classification_report(
            ytest, ytest_pred, output_dict=True)

        log_classification_report(report)
