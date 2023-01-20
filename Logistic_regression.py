import numpy as np


class L2LogisticRegression(object):
    def __init__(self, C=1):
        self.coef_ = None
        self.intercept_ = None
        self.C = C

    def sigmoid(self, t):
        return 1. / (1 + np.exp(-t))

    def basic_term(self, X, y, logits):
      
        # Вычисляет градиент логистической функции потерь по весам алгоритма 
        # (исключая регуляризационное слагаемое).
        grad = -X * (y * (1 - self.sigmoid(y * logits))).reshape(-1, 1)
        return grad.mean(axis=0)

    def regularization_term(self, weights):
      
        # Вычисляет регуляризационное слагаемое градиента функции потерь 
        # (без домножения на константу регуляриации).
        grad = 2*weights
        grad[0] = 0
        return grad

    def grad(self, X, y, logits, weights):
      
        # Принимает на вход X, y, logits и вычисляет градиент логистической 
        # функции потерь (включая регуляризационное слагаемое).
        grad = self.basic_term(X,y,logits) + self.C * self.regularization_term(weights)
        return grad

    def fit(self, X, y, max_iter=1000, lr=0.1):
      
        # Принимает на вход X, y и вычисляет веса по данной выборке.
        # Множество допустимых классов: {1, -1}.
        X = np.array(X)
        y = np.array(y)
        y = 2 * y - 1

        # Добавляем признак из единиц.
        X = np.hstack([np.ones([X.shape[0], 1]), X])

        l, n = X.shape
        # Инициализируем веса.
        weights = np.random.randn(n)

        losses = []

        for iter_num in range(max_iter):
            # Вычисляем градиент.
            logits = (X @ weights.reshape(n, 1)).ravel()  # [ell]
            grad = self.grad(X, y, logits, weights)
            # Обновляем веса.
            weights -= grad * lr

            # Вычисляем функцию потерь.
            loss = np.mean(np.log(1 + np.exp(-y * logits))) + self.C * np.sum(weights[1:] ** 2)
            losses.append(loss)

        # Объявляем окончательные веса.
        self.coef_ = weights[1:]
        self.intercept_ = weights[0]

        return losses

    def predict_proba(self, X):
        X = np.array(X)
        X = np.hstack([np.ones([X.shape[0], 1]), X])
        weights = np.concatenate([self.intercept_.reshape([1]), self.coef_])
        logits = (X @ weights.reshape(-1, 1))

        return self.sigmoid(logits)

    # Функция принимает на вход X и возвращает ответы модели.
    def predict(self, X, threshold=0.5):
        return self.predict_proba(X) >= threshold
