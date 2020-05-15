import numpy as np
from sklearn.preprocessing import LabelEncoder


class Dataset(object):
    def __init__(self, x=None, y=None):
        if isinstance(x, list):
            self.data = x
        elif isinstance(x, np.ndarray):
            self.data = x.tolist()

        if isinstance(y, np.ndarray):
            y = y.tolist()

        if y and len(y) > 0:
            if isinstance(y[0], str):
                enc = LabelEncoder()
                self.target = enc.fit_transform(y).tolist()
                self.labels = enc.classes_.tolist()
