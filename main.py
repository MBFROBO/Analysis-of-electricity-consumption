import tensorflow
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from include.plotter import Graph
from include.network import neural_network

# Корректируем данные, обучаем модель

if __name__ == '__main__':

    train = pd.read_csv('datasets/train.csv')
    train = train.drop('ID', axis= 1)
    test = pd.read_csv('datasets/test.csv')
    test = test.drop('ID', axis= 1)

    input_data = train['electricity_consumption']

    G = Graph(train, test, input_data)
    G.data_transform()
    train = G.data_revision()
    train, test, input_data = G.data_correct()
    G.correl()
    G.data_print()

    NN = neural_network(train, input_data, test)
    NN.coefficiets()
    NN.learn_model()
    NN.loss_plot()
    NN.accurancy_test()
    plt.show()