"""задание"""
#Измените функцию вычисления ошибки,
#чтобы она возвращала среднеквадратичное отклонение (RMSE).
#Как это повлияло на скорость обучения нейросети?
#Изменилось поведение при большом количестве итераций.
#Переобучения при 10^5 итерациях не произошло.

# Импорт библиотеки numpy для работы с массивами и математическими операциями
import numpy as np

# Инициализация начальных весов нейронной сети
# weights[0] - коэффициент для роста, weights[1] - коэффициент для веса
weights = np.array([0.2, 0.3])

# Функция нейронной сети - выполняет линейное преобразование
# inp - входные данные (массив роста и веса), weights - весовые коэффициенты
# Возвращает скалярное произведение (линейную комбинацию входов и весов)
def neural_networks(inp, weights):
    return inp.dot(weights)

# Измененная функция вычисления ошибки - теперь возвращает RMSE (Root Mean Square Error)
# true_prediction - целевое значение, prediction - предсказанное значение
# RMSE = sqrt(mean((true - pred)^2)) - среднеквадратичное отклонение
def get_error(true_prediction, prediction):
    return np.sqrt(np.mean((true_prediction - prediction) ** 2))

# Функция градиентного спуска для обучения нейронной сети
# inp - массив входных данных
# true_predictions - массив целевых значений
# weights - весовые коэффициенты
# learning_rate - скорость обучения
# epochs - количество эпох обучения
def gradient(inp, true_predictions, weights, learning_rate, epochs):
    # Цикл по количеству эпох обучения
    for i in range(epochs):
        error = 0  # Обнуление суммарной ошибки для текущей эпохи
        delta = np.zeros_like(weights)  # Создание нулевого массива для накопления градиентов
        
        # Проход по всем примерам обучающей выборки
        for j in range(len(inp)):
            current_inp = inp[j]  # Текущий входной пример (рост и вес)
            true_prediction = true_predictions[j]  # Целевое значение
            prediction = neural_networks(current_inp, weights)  # Предсказание нейросети
            
            # Накопление ошибки (с использованием новой функции RMSE)
            error += get_error(true_prediction, prediction)
            
            # Вывод отладочной информации
            print(
                "Prediction: %.10f, True_prediction: %.10f, Weights: %s"
                % (prediction, true_prediction, weights)
            )
            
            # Вычисление градиента для текущего примера
            # Обратите внимание: градиент вычисляется по-прежнему на основе MSE,
            # а не RMSE, так как производная RMSE сложнее
            delta += (prediction - true_prediction) * current_inp * learning_rate
        
        # Обновление весов: вычитаем средний градиент по всем примерам
        weights -= delta / len(inp)
        
        # Вывод суммарной ошибки для текущей эпохи
        print("Errors: %.10f" % error)
        print("-------------------")  # Разделитель между эпохами
    
    # Возвращаем обученные веса после всех эпох
    return weights

# Вспомогательная функция для вычисления вероятности
def calc_prob(person_h, person_w):
    return neural_networks(np.array([person_h, person_w]), weights)

# Обучающая выборка: массив пар [рост, вес]
inp = np.array(
    [
        [150, 40],  
        [140, 35],  
        [155, 45],  
        [185, 95],  
        [145, 40], 
        [195, 100], 
        [180, 95],  
        [170, 80],  
        [160, 90],  )
    ]
)

# Целевые значения: 0 = женщина, 100 = мужчина
true_predictions = np.array([0, 0, 0, 100, 0, 100, 100, 100, 100])

# Скорость обучения
learning_rate = 0.00001

# Количество эпох обучения
epochs = 10**5  # 100000 итераций

# Обучение нейронной сети
weights = gradient(inp, true_predictions, weights, learning_rate, epochs)

# Тестирование обученной модели
print(calc_prob(150, 45))  # Ожидается значение ближе к 0 (женщина)
print(calc_prob(170, 85))  # Ожидается значение ближе к 100 (мужчина)
