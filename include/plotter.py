from matplotlib import pyplot as plt
from pandas.plotting import autocorrelation_plot
from pandas import Series
import pandas as pd
import numpy as np
import datetime


class Graph:
    """
        Обрабатываем данные, строим графики
    """
    def __init__(self, data_train = None, data_test = None):
        self.train = data_train
        self.test = data_test

    def data_correct(self):
        """
            Исключим предикторы с минимальной корреляцией по выходному признаку
        """
        return self.train.drop('pressure', axis = 1), self.test.drop('pressure', axis = 1)
    
    def data_transform(self):
        """
            Преобразуем данные в более информативные (дата-время в часы, а в столбце var2, который, вероятно, представляет собой категорию потребителей), 
            принимая некоторые допущения, например,исклчив день/год/месяц. Это помешает видеть общую картину цикличности смен портебления электроэнергии.
        """

        time_train = self.train['datetime']
        index = 0
        print(len(time_train))
        print(len(self.train['datetime']))
        for t in time_train:
            time = int(t.split(' ')[1].split(':')[0])
            time_train[index] = time
            index += 1

        time_test = self.test['datetime']
        _index = 0
        for _t in time_test:
            _time = int(_t.split(' ')[1].split(':')[0])
            time_test[_index] = _time
            _index += 1
        
        self.train['var2'] = self.train['var2'].replace('A', 1)
        self.train['var2'] = self.train['var2'].replace('B', 2)
        self.train['var2'] = self.train['var2'].replace('C', 3)
 
        self.test['var2'] = self.test['var2'].replace('A', 1)
        self.test['var2'] = self.test['var2'].replace('B', 2)
        self.test['var2'] = self.test['var2'].replace('C', 3)
        
    def correl(self):
        """
            Находим корреляцию в данных между предикторами и выходной переменной на тренировочных данных
        """

        time = self.train['datetime']
        temp = self.train['temperature']
        var  = self.train['var1']
        press = self.train['pressure']
        wind = self.train['windspeed']
        var2 = self.train['var2']
        output = self.train['electricity_consumption']

        #Исследуем на сезонность:
        ### Присутствует слабая сезонная составляющая
        plt.figure('season output_research')
        autocorrelation_plot(output[:int(len(output)*0.5)])
 
        #Проверим зависимость выходной величины от предикторов (температура, вар1 ?, давление, скорость ветра, var2 ?)
        plt.figure('Температура')
        plt.plot(output[:int(len(output)*0.9)],temp[:int(len(output)*0.9)], '.')
        plt.xlabel('Выходная переменная')
        plt.ylabel('Температура')
        ### Прослеживается слабая завесимость от температуры

        plt.figure('Переменная 1')
        plt.plot(output[:int(len(output)*0.9)],var[:int(len(var)*0.9)], '.')
        plt.xlabel('Выходная переменная')
        plt.ylabel('Переменная 1')

        plt.figure('Давление')
        plt.plot(output[:int(len(output)*0.9)],press[:int(len(press)*0.9)], '.')
        plt.xlabel('Выходная переменная')
        plt.ylabel('Давление')

        plt.figure('Скорость ветра')
        plt.plot(output[:int(len(output)*0.9)],wind[:int(len(wind)*0.9)], '.')
        plt.xlabel('Выходная переменная')
        plt.ylabel('Скорость ветра')

        plt.figure('Переменная 2')
        plt.plot(output[:int(len(output)*0.9)],var2[:int(len(var2)*0.9)], '.')
        plt.xlabel('Выходная переменная')
        plt.ylabel('Переменная 2')

        
    def data_print(self):

        print('-------------Train data---------------')
        print(self.train.tail(5))
        print('-------------End data-----------------')
        print('-------------Test data---------------')
        print(self.test.tail(5))
        print('-------------End data-----------------')