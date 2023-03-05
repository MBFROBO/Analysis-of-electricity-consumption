import tensorflow
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from include.plotter import Graph

if __name__ == '__main__':

    train = pd.read_csv('datasets/train.csv')
    train = train.drop('ID', axis=1)
    test = pd.read_csv('datasets/test.csv')
    test = test.drop('ID', axis= 1)

    G = Graph(train, test)
    G.data_transform()
    G.correl()
    G.data_print()
    train, test = G.data_correct()
    plt.show()