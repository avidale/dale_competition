import numpy as np
import sys
from collections import deque

from sklearn.linear_model import Ridge, LinearRegression
from numpy.linalg import solve

if sys.version_info.major == 2:
    input = raw_input

OFFSET = 3000
SKIP_SIZE = 28
TARGET_CHANNEL = 15
N_CHANNELS = 21

experiment_id = input()


class DelayedRLSPredictor:
    def __init__(self, n_channels, M=3, lambda_=0.999, delta=100, delay=0, mu=0.3):
        self._M = M
        self._lambda = lambda_
        self._delay = delay
        self._mu = mu
        size = M * n_channels
        self._w = np.zeros((size,))
        self._P = delta * np.eye(size)
        self.regressors = [] #deque(maxlen=M + delay + 1)
    
    def append(self, sample):
        self.regressors.append(sample)
    def update(self):
        regressors = np.array(self.regressors)
        if regressors.shape[0] > self._delay + self._M:
            # predicted var x(t) 
            # это ПОСЛЕДНЯЯ строка, которую я получил. 
            predicted = regressors[-1, TARGET_CHANNEL]

            # predictor var [x(t - M), x(t - M + 1), ..., x(t - delay)]
            # читаю M последних строк (с учётом лага) в один вектор длины M * n_channels
            predictor = regressors[- self._M - self._delay - 1: - self._delay - 1].flatten()  #

            # update helpers
            pi = np.dot(predictor, self._P) # это вектор-строка - произведение вектора и матрицы XP
            # k = XP/(l + XPy) # нормируем полученное произведение
            k = pi / (self._lambda + np.dot(pi, predictor)) 
            # обновляем матрицу, вычтя матричное произведение векторов k*pi P = 1/l * (P -k*pi)
            self._P = 1 / self._lambda * (self._P - np.dot(k[:, None], pi[None, :]))

            # update weights
            dw = (predicted - np.dot(self._w, predictor)) * k
            self._w = self._w + self._mu * dw
            
    def predict_linear(self):
        regressors = np.array(self.regressors)
        if regressors.shape[0] > self._delay + self._M:
            # return prediction x(t + delay)
            return np.dot(self._w, regressors[- self._M:].flatten())
        # if lenght of regressor less than M + delay + 1 return 0
        return 0 
    def update_predict(self, sample):
        self.append(sample)
        self.update()
        return self.predict_linear()
    def total_update(self):
        self.total_update_ridge()
    def total_update_ridge(self):
        regressors = np.array(self.regressors)
        X = []
        y = []
        # i символизирует первую строку в X
        # X будет использовать строки [i, i+M)
        # y будет использовать строку i + delay + M
        for i in range(0, regressors.shape[0] - self._delay - self._M):
            predictor = regressors[- self._M - self._delay - 1: - self._delay - 1].flatten()
            X.append(predictor)
            y.append(regressors[i + self._delay + self._M, TARGET_CHANNEL])
            _ = 1
        X = np.array(X)
        y = np.array(y)
        
        #self.ridge = Ridge(alpha = 0.000001)
        #self.ridge = LinearRegression() #Ridge(solver = 'lsqr', max_iter = 1)
        
        #print(regressors.shape)
        #print(X.shape, y.shape)
        #self.ridge.fit(X,y)
        #print(X.shape[1])
        self.beta = solve(np.dot(X.transpose(), X) + np.eye(X.shape[1])*0.0003, np.dot(X.transpose(), y))
        #print(self.ridge.coef_)
    def ridge_predict(self):
        #regressors = np.array(self.regressors) # 
        predictor = [v for x in self.regressors[- self._M - self._delay - 1: - self._delay - 1] for v in x]
        #predictor = [regressors[- self._M - self._delay - 1: - self._delay - 1].flatten()]
        #prediction = self.ridge.predict([predictor]    )[0]
        prediction = np.dot(predictor, self.beta)
        #print(type(prediction))
        return prediction
        
    def append_predict(self, sample):
        """
        Устаревший метод, который делает всё в один мах - и добавляет наблюдения, и дообучается, и предсказывает.
        Вместо него следует использовать update_predict, который делает три вещи отдельными функциями. 
        """
        # метод добавляет sample в правый конец очереди 
        self.regressors.append(sample)
        regressors = np.array(self.regressors)
        if regressors.shape[0] > self._delay + self._M:
            # predicted var x(t) 
            # это ПОСЛЕДНЯЯ строка, которую я получил. 
            predicted = regressors[-1, TARGET_CHANNEL]

            # predictor var [x(t - M), x(t - M + 1), ..., x(t - delay)]
            # читаю M последних строк (с учётом лага) в один вектор длины M * n_channels
            predictor = regressors[- self._M - self._delay - 1: - self._delay - 1].flatten()  #

            # update helpers
            pi = np.dot(predictor, self._P) # это вектор-строка - произведение вектора и матрицы XP
            # k = XP/(l + XPy) # нормируем полученное произведение
            k = pi / (self._lambda + np.dot(pi, predictor)) 
            # обновляем матрицу, вычтя матричное произведение векторов k*pi P = 1/l * (P -k*pi)
            self._P = 1 / self._lambda * (self._P - np.dot(k[:, None], pi[None, :]))

            # update weights
            dw = (predicted - np.dot(self._w, predictor)) * k
            self._w = self._w + self._mu * dw

            # return prediction x(t + delay)
            return np.dot(self._w, regressors[- self._M:].flatten())

        # if lenght of regressor less than M + delay + 1 return 0
        return 0


rls = DelayedRLSPredictor(n_channels=N_CHANNELS, M=5, lambda_=0.9999, delta=0.01, delay=SKIP_SIZE, mu=1)

# читаю первые 3000 строк, обучающие, и каждый раз что-то предсказываю (и забываю)
for i in range(OFFSET):
    cur_data = list(map(float, input().split()))
    #pr0 = rls.update_predict(cur_data)
    rls.append(cur_data)
    rls.update()
    
rls.total_update_ridge()
pr0 = rls.predict_linear()
pr1 = rls.ridge_predict()
w1 = 0.1
prediction = pr0 * (1-w1) + pr1 * w1
# вывожу предсказание после первых 3000 строк
print(prediction)
sys.stdout.flush()

# читаю очередную тестовую строку и делаю предсказание на её основе. 
while True:
    cur_data = list(map(float, input().split()))
    #prediction = rls.update_predict(cur_data)
    rls.append(cur_data)
    rls.update()
    pr0 = rls.predict_linear()
    pr1 = rls.ridge_predict()
    prediction = pr0 * (1-w1) + pr1 * w1
    print(prediction)
    sys.stdout.flush()


