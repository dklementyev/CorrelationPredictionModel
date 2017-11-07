import numpy as np
import pandas as pd
from sklearn import cross_validation as cv
from sklearn.metrics.pairwise import pairwise_distances
import warnings
from math import sqrt
from sklearn.metrics import mean_squared_error
#Заигнорить numpy Варнинги
np.seterr(divide='ignore', invalid='ignore')

header = ['user_id','item_id','rating','timestamp']

df = pd.read_csv('ml-100k/u.data', sep='\t', names=header)

#Кол-во (уникальных) юзеров и айтемов
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]

#Датасет для обучения - 75%, теста 25.
train_data, test_data = cv.train_test_split(df,test_size=0.25)

#Две юзер-айтем матрицы - для трэйнинга и для теста
train_data_matrix = np.zeros((n_users,n_items))
for line in train_data.itertuples():
    train_data_matrix[line[1]-1,line[2]-1] = line[3]

test_data_matrix = np.zeros((n_users,n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1,line[2]-1] = line[3]

#Используем Cousine Similarity для нахождения близости
user_similarity = pairwise_distances(train_data_matrix,metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')

#Функция предикта (предсказания)
def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        ratings_diff = (ratings - mean_user_rating[:,np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
         pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

#Функция среднеквадратичной ошибки
def rmse(prediction, actual):
    prediction = prediction[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction,actual))

#Получаем предикты
item_predict = predict(train_data_matrix, item_similarity, type='item')
user_predict = predict(train_data_matrix, user_similarity, type='user')

print(item_predict)
print('\n')
print(user_predict)

#Вычисляем ошибку
item_based_error = rmse(item_predict,test_data_matrix)
user_based_error = rmse(user_predict,test_data_matrix)

print('Item Based Error: ' + str(item_based_error))
print('User Based Error: ' + str(user_based_error))


