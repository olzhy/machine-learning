#!/usr/bin/env python3
import perceptron
import csv

if '__main__' == __name__:
    # perceptron instance
    p = perceptron.Perceptron(eta=1.0, max_iter=100)

    # training
    x = []
    y = []
    with open('iris.data') as f:
        for sample in csv.reader(f):
            x.append([float(i) for i in sample[:4]])
            y.append(1 if sample[4] == 'Iris-setosa' else -1)
    p.fit(x, y)

    # predict
    with open('predict.data') as f:
        for sample in csv.reader(f):
            xi, yi = [float(i) for i in sample[:4]], sample[4]
            print('predict label: {}, real: {}'.format(p.predict(xi), yi))
