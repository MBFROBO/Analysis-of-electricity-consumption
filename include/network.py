import numpy as np
import pandas as pd
import tensorflow as tf

from keras.optimizers import Adam


class neural_network:

    def __init__(self, train, input_data):

        """
            Инициализация параметров для нейронной сети
        """
        self.epoch = 100        # Количество эпох обучения
        self.bathes = 50        # Размер входного пакета
        self.num_input = 4      # Число нейронов входного слоя
        self.hidden_1 = 25      # Число нейронов первого скрытого слоя
        self.hidden_2 = 30      # Число нейронов второго скрытого слоя
        self.out_layer = 1      # Число нейронов на выходном слое
        self.train = train
        self.x = input_data

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
        hidden_layer1 = tf.add(tf.matmul(_input, self.weigth[0]), self.bias[0])
        hidden_layer1 = tf.nn.sigmoid(hidden_layer1)    # Сигмоидная функция активации 1/(1 - e^-x)

        hidden_layer2 = tf.add(tf.matmul(hidden_layer1, self.weigth[1]), self.bias[1])
        hidden_layer2 = tf.keras.activations.relu(hidden_layer2)    # Функция активации relu. при x < 0 = 0, при x >= 0 -> x

        out_layer = tf.add(tf.matmul(hidden_layer2, self.weigth[2]), self.bias[2])
        out_layer = tf.keras.activations.linear(out_layer) # Линейная y = x

        return out_layer
    