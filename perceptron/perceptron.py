from typing import List
import random


class Perceptron:
    """
    Perceptron binary classifier
    """
    def __init__(self, eta: float = 1.0, max_iter: int = 1000):
        """
        Perceptron constructor
        :param eta: 0 < eta <= 1.0
        :param max_iter: max iteration
        """
        self.eta = eta
        self.max_iter = max_iter

        random.seed(1)

    def fit(self, x: List[List[float]], y: List[int]) -> None:
        """
        Fit is used for data training
        after the execution, _w (weight vector) will be produced
        :param x: input, vector list
        :param y: output, class label list
        :return:  None
        """
        if len(x) <= 0:
            return

        # initialize weight vector to random integers
        self._w = [random.randint(-10, 10) for _ in range(len(x[0]) + 1)]

        times = 0
        while times < self.max_iter:
            times += 1
            errors = 0
            for xi, yi in zip(x, y):
                y_predict = self.predict(xi)
                if yi - y_predict != 0:
                    errors += 1
                    for i in range(len(xi)):  # update vector _w
                        self._w[i + 1] += self.eta * (yi - y_predict) * xi[i]
                    self._w[0] += self.eta * (yi - y_predict)
                print(
                    'times: {}, xi: {}, yi: {}, y_predict: {}, _w: {}'.format(
                        times, xi, yi, y_predict, self._w))
            if 0 == errors:
                break

    def _predict(self, xi: List[float]) -> float:
        """
        Calculate the predictive value for a single sample input xi
        :param x: a single sample input xi
        :return: dot product of vector _w and xi
        """
        return sum([self._w[i + 1] * xi[i]
                    for i in range(len(xi))]) + self._w[0]

    def predict(self, xi: List[float]) -> int:
        """
        Predict xi belongs to class +1 or -1
        :param xi: a single sample input xi
        :return: class +1 or -1
        """
        return 1 if self._predict(xi) >= 0 else -1
