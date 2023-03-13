import numpy as np
import pandas as pd
import tensorflow as tf
import random

from matplotlib import pyplot as plt
from pandas.plotting import autocorrelation_plot
from keras import optimizers


class neural_network:

    def __init__(self, train, input_data, test_data):

        """
            Инициализация параметров для нейронной сети
        """
        self.epoch = 50      # Количество эпох обучения
        self.bathes = 50        # Размер входного пакета
        self.num_input = 5     # Число нейронов входного слоя
        self.hidden_1 = 25      # Число нейронов первого скрытого слоя
        self.hidden_2 = 25      # Число нейронов второго скрытого слоя
        self.out_layer = 1      # Число нейронов на выходном слое
        self.x = train
        self.y = input_data
        self.test = test_data

        self.optimizer = optimizers.Adam(0.01)
        tf.random.set_seed(1)

    def coefficiets(self):

        """
            Генерируем коэфициенты w, b для нейронов двух скрытых, входного и выходного слоёв.
            Функция в каждом нейроне представляется линейной w*x + b, где x = input (вводимое значение)
        """

        self.weights = [
            tf.Variable(tf.random.normal([self.num_input, self.hidden_1])),
            tf.Variable(tf.random.normal([self.hidden_1, self.hidden_2])),
            tf.Variable(tf.random.normal([self.hidden_2, self.out_layer]))
        ]

        self.bias = [
            tf.Variable(tf.random.normal([self.hidden_1])),
            tf.Variable(tf.random.normal([self.hidden_1])),
            tf.Variable(tf.random.normal([self.out_layer]))
        ]

        with open('include/coefficients.txt', 'w') as file:
            file.write('Weights\n\r')
            file.write(f'{self.weights}')
            file.write('\n\rBias\n\r')
            file.write(f'{self.bias}')
    
    def model(self, _input):
        """
            Создаём концептуальную модель нейронной сети
        """
        hidden_layer1 = tf.add(tf.matmul(_input, self.weights[0]), self.bias[0])
        hidden_layer1 = tf.nn.sigmoid(hidden_layer1)    # Сигмоидная функция активации 1/(1 - e^-x)

        hidden_layer2 = tf.add(tf.matmul(hidden_layer1, self.weights[1]), self.bias[1])
        hidden_layer2 = tf.keras.activations.relu(hidden_layer2)    # Функция активации relu. при x < 0 = 0, при x >= 0 -> x

        out_layer = tf.add(tf.matmul(hidden_layer2, self.weights[2]), self.bias[2])
        out_layer = tf.keras.activations.linear(out_layer) # Линейная y = x

        return out_layer
    
    def loss(self):
        return tf.keras.losses.mean_absolute_error(self.y_bath, self.model(self.X_bath))
    
    def learn_model(self):

        self.numpy_array = []
        for ep in range(self.epoch):
            for i in range(0, int(len(self.x) / self.bathes -1)):
                self.X_bath = self.x[i *self.bathes:(i + 1) * self.bathes]
                self.y_bath = self.y[i*self.bathes:(i + 1) * self.bathes]
                self.optimizer.minimize(self.loss, [self.weights, self.bias])
                # tensArray = tensArray.write(n, loss())
            self.X_bath = self.x[(i + 1) * self.bathes:]
            self.y_bath = self.y[(i + 1) * self.bathes:]
            self.optimizer.minimize(self.loss, [self.weights, self.bias])
            self.numpy_array.append(self.loss()[0].numpy())
            print('Epoch: ' + str(ep+1), ' | ','loss: ', str(self.loss()[0]))

    def loss_plot(self):

        fig = plt.figure('Loss')
        fig.set_label('Loss')
        fig.legend('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(self.numpy_array)

    def accurancy_test(self):
        results = self.model(self.test)
        
        plt.figure('season output_research')
        autocorrelation_plot(results.numpy()[:int(len(results.numpy()))])

        plt.figure('Выходные значения')
        plt.plot(results.numpy())
