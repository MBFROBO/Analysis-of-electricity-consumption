import pandas as pd
import numpy as np
import datetime

from matplotlib import pyplot as plt
from pandas.plotting import autocorrelation_plot
from pandas import Series, DataFrame

class Graph:

    """
        Обрабатываем данные, строим графики
    """
    def __init__(self, data_train = None, data_test = None):
        self.train = data_train                                                                             # тренировочные данные
        self.test = data_test                                                                               # Тестовые данные
        self.names = ['datetime','temperature','var1','pressure','windspeed', 'var2']                       # Список колонок тренировочных данных

        self.high = []  # Верхний квантильный уровень
        self.Low = []   # Нижний квантильный уровень
        self.quantiles = {}
    def data_correct(self):
        """
            Исключим предикторы с минимальной корреляцией по выходному признаку
        """
        return self.train.drop('pressure', axis = 1), self.test.drop('pressure', axis = 1)
    

    def data_revision(self):
        """
            Расщепляя каждый массив данных на 4 части, проведём квантильный анализ на выбросы
        """
        for i in self.names:
            if i == 'datetime':
                continue
            
            else:
                try:

                    data_frame_1 = self.train[i][int(len(self.train[i])*0.00):int(len(self.train[i])*0.25)]
                    data_frame_2 = self.train[i][int(len(self.train[i])*0.25):int(len(self.train[i])*0.50)]
                    data_frame_3 = self.train[i][int(len(self.train[i])*0.50):int(len(self.train[i])*0.75)]
                    data_frame_4 = self.train[i][int(len(self.train[i])*0.75):int(len(self.train[i])*1.00)]

                    data_array = [data_frame_1, 
                                data_frame_2,
                                data_frame_3,
                                data_frame_4]
                    
                    color = ['r','k','g','c']
                    for j,c in zip(data_array, color):
                        """
                            Использую метод квантильного отсеивания
                        """
                            # Для некоторых типов коррелируемых данных (которые возможно обработать таким типом), сделаем отсеивание по верхнему и нижнему уровням
                        if i == 'pressure' or i == 'windspeed':
                            j = j.astype('float')
                            output_quantile_75 = np.percentile(np.log(np.array(j)), 75, method='normal_unbiased')
                            output_quantile_25 = np.percentile(np.log(np.array(j)), 25, method= 'normal_unbiased')
                            IQR = output_quantile_75 - output_quantile_25

                            self.high.append(float(output_quantile_75) + 1.1*float(IQR))
                            self.Low.append(float(output_quantile_25) - 1.1*float(IQR))

                            self.quantiles[i] = [self.high, self.Low]
                except KeyError:
                    pass
                    
        try:
            self.train = self.train.astype('float')

            print(self.train)
            print('--------------------Quantiles -------------------')
            print(self.quantiles)
            print('-------------------end quantiles------------------')

            bad_index_high_press = self.train.index[np.log(np.array(self.train['pressure'])) > max(self.quantiles['pressure'][0])].tolist()
            print('Индексы для удаления по верхней границе,длина: ', len(bad_index_high_press))
            bad_index_low_press = self.train.index[np.log(np.array(self.train['pressure'])) < np.mean(self.quantiles['pressure'][1])].tolist()
            print('Индексы для удаления по нижней границе, длина: ', len(bad_index_low_press))

            self.train.drop(bad_index_high_press)
            self.train.drop(bad_index_low_press)

            bad_index_high_wind = self.train.index[np.log(np.array(self.train['windspeed'])) > np.mean(self.quantiles['windspeed'][0])].tolist()
            print('Индексы для удаления по верхней границе,длина: ', len(bad_index_high_wind))
            bad_index_low_wind = self.train.index[np.log(np.array(self.train['windspeed'])) < min(self.quantiles['windspeed'][1])].tolist()
            print('Индексы для удаления по нижней границе, длина: ', len(bad_index_low_wind))
            
            self.train.drop(bad_index_high_wind)
            self.train.drop(bad_index_low_wind)
            # Сбрасываем индексы
            self.train.reset_index(drop= True , inplace= True)
            # Строим квантильные прямые
            plt.figure('pressure')

            plt.axhline(np.mean(self.quantiles['pressure'][1]), xmin = 0, xmax = len(j), color = c)
            plt.axhline(max(self.quantiles['pressure'][0]), xmin = 0, xmax = len(j), color = c)

            plt.figure('windspeed')

            plt.axhline(min(self.quantiles['windspeed'][1]), xmin = 0, xmax = len(j), color = c)
            plt.axhline(np.mean(self.quantiles['windspeed'][0]), xmin = 0, xmax = len(j), color = c)

            return self.train
        
        except IndexError as e:
            print('-----------------------------error---------------------------')
            print(e)
            print('-----------------------------error---------------------------')
        except KeyError as e:
            pass
            

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
    

        ### Выполним нормализацию данных для повышения точности в обучении ###
        self.train = self.train.drop('electricity_consumption', axis = 1)
        self.train = self.train.drop('var2', axis = 1)
        self.test = self.test.drop('var2', axis = 1)
        
        self.train = (self.train - self.train.min()) / (self.train.max() - self.train.min()) + 1
        self.test = (self.test - self.test.min()) / (self.test.max() - self.test.min()) + 1


    
    def correl(self):
        """
            Находим корреляцию в данных между предикторами и выходной переменной на тренировочных данных
        """
        try:
            # time = np.array(self.train['datetime'])
            temp = np.array(self.train['temperature']).astype('float')
            var  = np.array(self.train['var1']).astype('float')
            press = np.array(self.train['pressure']).astype('float')
            wind = np.array(self.train['windspeed']).astype('float')
            # output = (self.train['electricity_consumption']).astype('float')
            

            # self.train['var2'] = self.train['var2'].replace('A', 1)
            # self.train['var2'] = self.train['var2'].replace('B', 2)
            # self.train['var2'] = self.train['var2'].replace('C', 3)
    
            # self.test['var2'] = self.test['var2'].replace('A', 1)
            # self.test['var2'] = self.test['var2'].replace('B', 2)
            # self.test['var2'] = self.test['var2'].replace('C', 3)

            # var2 = (self.train['var2']).astype('float')
            # Исследуем на сезонность:
            ## Присутствует слабая сезонная составляющая
            # plt.figure('season output_research')
            # autocorrelation_plot(output[:int(len(output)*0.5)])
    
            #Проверим зависимость выходной величины от предикторов (температура, вар1 ?, давление, скорость ветра, var2 ?)
            # plt.figure(self.names[1])
            # plt.plot(np.log(temp[:int(len(output))]), '.')
            # plt.xlabel('Выходная переменная')
            # plt.ylabel('Температура')
            # ### Прослеживается слабая завесимость от температуры

            # plt.figure(self.names[2])
            # plt.plot(np.log(var[:int(len(var))]), '.')
            # plt.xlabel('Выходная переменная')
            # plt.ylabel('Переменная 1')

            plt.figure(self.names[3])
            plt.plot(np.log(press[:int(len(press))]), '.')
            plt.xlabel('Выходная переменная')
            plt.ylabel('Давление')
            #Хорошая зависимость
            plt.figure(self.names[4])
            plt.plot(np.log(wind[:int(len(wind))]), '.')
            plt.xlabel('Выходная переменная')
            plt.ylabel('Скорость ветра')

            # plt.figure(self.names[5])
            # plt.plot(np.log(var2[:int(len(var2))]), '.')
            # plt.xlabel('Выходная переменная')
            # plt.ylabel('Переменная 2')

            # plt.figure(self.names[6])
            # plt.plot(np.log(output[:int(len(var2))]), '.')
            # plt.xlabel('Выходная переменная')
            # plt.ylabel('Потребление')
        except Exception as e:
            print(e)
            pass
        
    def data_print(self):

        print('-------------Train data---------------')
        print(self.train.tail(5))
        print('-------------End data-----------------')
        print('-------------Test data---------------')
        print(self.test.tail(5))
        print('-------------End data-----------------')