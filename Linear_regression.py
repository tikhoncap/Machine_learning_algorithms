import numpy as np


# Точный аналитический метод
class LinearRegression:
    
    def __init__(self):
        
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
      
        # Функция обучения линейной регрессии.
        # Функция принимает на вход обучающую выборку 
        # (Х — матрица признаков, y — массив ответов, значений целевой переменной),
        # и подбирает коэффициенты линейной регрессии.
        
        # Входящие параметры: 
        # X: матрица размера (n, f), где n — количество элементов датасета, 
        # f — количество признаков
        # y: массив размера (n, ), где n — количество элементов датасета

        X = np.array(X)
        y = np.array(y)

        X = np.hstack((np.ones((X.shape[0], 1)), X))
        k = np.linalg.inv(X.T @ X) @ X.T @ y
        
        # Список коэффициентов, которые модель поставила 
        # в соответствие признакам датасета. 
        self.coef_ = k[1:]
        # Коэффициент — свободный член.
        self.intercept_ = k[0]
        
    def predict(self, X):

        # Функция получения предсказания линейной регрессии по входящему массиву признаков Х.  
        
        # Входящие параметры: 
        # X: матрица размера (n, f), где n — количество элементов датасета, 
        # f — количество признаков
        
        y_pred = X @ self.coef_ + self.intercept_   
        return y_pred
      

# Приближенный численный метод
class LinearRegression2:
    
    # Инициализируем lr: "темп обучения", epochs: количество итераций, 
    # weights, bias: параметры
    def __init__(self, lr=0.01, epochs=800):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None
        
    # Обучающая функция
    def fit(self, X, y):
      
        # Размер X: (количество обучающих образцов: m, количество    
        # признаков: n)
        m, n = X.shape    
    
        # Инициализируем веса в виде матрицы нулей размера: (число
        # признаков: n, 1) и смещение 0
        self.weights = np.zeros((n,1))
        self.bias = 0
        
        y = y.reshape(m,1)
        
        losses = []
        
        # Алгоритм градиентного спуска:
        for epoch in range(self.epochs):
        
            # Вычисляем предсказанное значение
            y_hat = np.dot(X, self.weights) + self.bias
     
            # Вычисляем функцию потерь
            loss = np.mean((y_hat - y)**2)
            losses.append(loss)
    
            # Вычисляем производные параметров(веса и смещение) 
            dw = (1/m)*np.dot(X.T, (y_hat - y))
            db = (1/m)*np.sum((y_hat - y))
            
            #Обновляем параметры
            self.weights -= self.lr*dw
            self.bias -= self.lr*db
        
        return self.weights, self.bias, losses

    #Функция предсказания
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
