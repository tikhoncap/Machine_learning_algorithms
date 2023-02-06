import numpy as np

# Функция, вычисляющая евклидово расстояние
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


class KNeighborsClassifier:
    def __init__(self, k=3):
        self.k = k

    # Функция принимает на вход обучающую выборку 
    # (Х — матрица признаков, y — массив ответов, значений целевой переменной)
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # Функция получения предсказания по входящему массиву признаков Х
    def predict(self, X):
        y_pred = np.array([self._predict(x) for x in X])
        return y_pred

    def _predict(self, x):
      
        # Вычисляем все расстояния между входным вектором и всеми векторами в обучающем массиве
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Сортируем вектора по расстоянию в порядке возрастания и отбираем первые k векторов
        k_indexes = np.argsort(distances)[: self.k]
        k_neighbor_labels = [self.y_train[i] for i in k_indexes]
        
        # Возвращаем метку самого часто встречающегося класса
        frequent = max(set(k_neighbor_labels), key=k_neighbor_labels.count)
        return frequent
