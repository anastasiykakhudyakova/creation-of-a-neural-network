# Tensor - основной класс для работы с тензорами и автоматическим дифференцированием
"""тензоры и автоматическая дифференциация
Логика предыдущих работ заключалась в работе с нейронными сетями из нескольких слоев.
Матричном перемножении весовых коэффициентов на значения нейронов при определении прогнозов в прямом распространении вычислений. 
На этом этапе использовались разные функци активации. 
Далее в обратном распространении послойно вычисляли дельты (градиент слоя) с учетом производных используемых функций активации, начиная с выходного слоя. 
Производили перерасчет матриц весовых коэффициентов.
В своих расчетах мы использовали три-четыре слоя. Но на практике часто бывают нейронные сети использующие 100 слоев.
Для реализации подобных вычислений уйдет немало ресурсов и скорее всего мы получим множество ошибок.
Конечно уже сеть такие фреймворки как tensorFlow или PyTorch
Разработав фреймворк, мы сможем значительно сократить и упростить код, реализующий обучение нейросети с функцией активации softmax. 
Также как облегчило работу использование библиотеки numpy.

Тензор – это очень простая вещь, по сути тоже самое, что вектор или матрица.
Просто тензором принято называть векторы и матрицы, когда речь идет о программировании/создании нейросетей.
"""

#Добавление автоматической дифференциации-это  autograd Это ключевой механизм современных фреймворков глубокого обучения (PyTorch, TensorFlow).
#Суть: автоматическое вычисление градиентов (производных) сложных функций без ручного вывода формул.
#Как работает:Запоминает все операции в графе вычислений
#При прямом проходе вычисляет значения
#При обратном проходе применяет цепное правило для вычисления градиентов


"""Отрицание: Проверяется корректность вычисления градиента при использовании унарного минуса в сложном выражении.
Вычитание: Проверяется правильность распространения градиента через операцию вычитания.
Умножение: Проверяется корректность правила произведения при обратном распространении.
Сложение: Проверяется базовое сложение и распределение градиента.
Сумма и расширение: Демонстрируются дополнительные операции и их взаимодействие с автоградиентом.
Каждая проверка включает:
Создание тензоров с autograd=True
Построение вычислительного графа
Вызов backward() для запуска обратного распространения
Вывод градиентов для визуальной проверки правильности"""
import numpy as np
class Tensor(object):
    
    # Счетчик для присвоения уникальных идентификаторов каждому тензору
    id_count = 0

    # Конструктор класса Tensor
    def __init__(self, data, creators=None, operation_on_creation=None, autograd=False, id=None):
        # Вызов метода инициализации
        self.init(data, creators, operation_on_creation, autograd, id)

    # Метод инициализации тензора
    def init(self, data, creators=None, operation_on_creation=None, autograd=False, id=None):
        # Преобразование данных в массив numpy для эффективных вычислений
        self.data = np.array(data)
        # Ссылки на тензоры-родители, которые создали этот тензор
        self.creators = creators
        # Операция, которая создала этот тензор (для обратного распространения)
        self.operation_on_creation = operation_on_creation
        # Градиент тензора (изначально None)
        self.grad = None
        # Флаг, включающий автоматическое дифференцирование для этого тензора
        self.autograd = autograd
        # Словарь для отслеживания количества детей для каждого родителя
        self.children = {}
        
        # Присвоение уникального идентификатора, если не указан
        if id is None:
            self.__class__.id_count += 1
            self.id = self.__class__.id_count
            
        # Если есть создатели, регистрируем этот тензор как их ребенка
        if self.creators is not None:
            for creator in creators:
                if self.id not in creator.children:
                    creator.children[self.id] = 1
                else:
                    creator.children[self.id] += 1

    # Метод обратного распространения градиента
    def backward(self, grad=None, grad_origin=None):
        # Проверяем, включено ли автоматическое дифференцирование
        if self.autograd:
            # Если градиент не передан, инициализируем единичным тензором
            if grad is None:    
                grad = Tensor(np.ones_like(self.data))
            # Если указан источник градиента (ребенок)
            if grad_origin is not None:
                # Уменьшаем счетчик ожидаемых градиентов от этого ребенка
                if self.children[grad_origin.id] > 0:
                    self.children[grad_origin.id] -= 1
                            
        # Инициализируем или аккумулируем градиент
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad
        # Продолжаем обратное распространение, если все дети отправили градиенты
        if self.creators is not None and (self.check_grads_from_children() or grad_origin is None):
                # Обработка операции сложения
                if self.operation_on_creation == "+":
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad, self)
                    
                # Обработка операции отрицания (унарный минус)
                elif self.operation_on_creation == "-1":
                    self.creators[0].backward(self.grad.__neg__(), self)
                    
                # Обработка операции вычитания
                elif self.operation_on_creation == "-":
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad.__neg__(), self)
                    
                # Обработка операции умножения (поэлементного)
                elif self.operation_on_creation == "*":
                    new_grad = self.grad * self.creators[1]
                    self.creators[0].backward(new_grad, self)
                    new_grad = self.grad * self.creators[0]
                    self.creators[1].backward(new_grad, self)
                    
                # Обработка операции суммирования по оси
                elif "sum" in self.operation_on_creation:
                    axis = int(self.operation_on_creation.split("_")[1])
                    self.creators[0].backward(self.grad.expand(axis, self.creators[0].data.shape[axis]), self)
                    
                # Обработка операции расширения по оси
                elif "expand" in self.operation_on_creation:
                    axis = int(self.operation_on_creation.split("_")[1])
                    self.creators[0].backward(self.grad.sum(axis), self)
                    
                # Обработка операции транспонирования
                elif self.operation_on_creation == "transpose":
                    self.creators[0].backward(self.grad.transpose(), self)
                    
                # Обработка операции матричного умножения (dot product)
                elif self.operation_on_creation == "dot":
                    temp = self.grad.dot(self.creators[1].transpose())
                    self.creators[0].backward(temp, self)
                    temp = self.creators[0].transpose().dot(self.grad)
                    self.creators[1].backward(temp, self)

                # Обработка сигмоидальной функции активации
                elif self.operation_on_creation == "sigmoid":
                    temp = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad * (self*(temp - self)), self)

                # Обработка гиперболического тангенса
                elif self.operation_on_creation == "tanh":
                    temp = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad * (temp - self * self), self)

                # Обработка функции активации ReLU
                elif self.operation_on_creation == "relu":
                    temp = self.grad * Tensor((self.creators[0].data > 0) * 1.0)
                    self.creators[0].backward(temp, self)
                                    
    # Проверка, все ли дети отправили градиенты
    def check_grads_from_children(self):
        for id in self.children:
            if self.children[id] != 0:
                return False
        return True
            
    # Метод сложения тензоров
    def add(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data + other.data, [self, other], "+", True)
        return Tensor(self.data + other.data)

    # Перегрузка оператора сложения (+)
    def __add__(self, other):
        return self.add(other)
    
    # Преобразование тензора в строку для вывода
    def __str__(self):
        return str(self.data)

    # Перегрузка оператора вычитания (-)
    def __sub__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data - other.data, [self,other], "-", True)
        return Tensor(self.data - other.data)

    # Перегрузка унарного оператора отрицания (-)
    def __neg__(self):
        if self.autograd:
            return Tensor(self.data * (-1), [self], "-1", True)
        return Tensor(self.data * (-1))

    # Перегрузка оператора умножения (*)
    def __mul__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data * other.data, [self, other], "*", True)
        return Tensor(self.data * other.data)

    # Метод суммирования по оси
    def sum(self, axis):
        if self.autograd:
            return Tensor(self.data.sum(axis), [self], "sum_" + str(axis), True)
        return Tensor(self.data.sum(axis))
    
    # Метод расширения тензора по оси
    def expand(self, axis, count_copies):
        # Создание порядка осей для транспонирования
        transpose = list(range(0, len(self.data.shape)))
        transpose.insert(axis, len(self.data.shape))
        # Формирование новой формы тензора
        expand_shape = list(self.data.shape) + [count_copies]
        # Повторение данных и изменение формы
        expand_data = (self.data.repeat(count_copies).reshape(expand_shape))
        # Транспонирование для правильного расположения осей
        expand_data = expand_data.transpose(transpose)
        if self.autograd:
            return Tensor(expand_data, [self], "expand_" + str(axis), True)
        return Tensor(expand_data)

    # Метод матричного умножения (dot product)
    def dot(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data.dot(other.data), [self, other], "dot", True)
        return Tensor(self.data.dot(other.data))

    # Метод транспонирования тензора
    def transpose(self):
        if self.autograd:
            return Tensor(self.data.transpose(), [self], "transpose", True)
        return Tensor(self.data.transpose())

    # Сигмоидальная функция активации
    def sigmoid(self):
        if self.autograd:
            return Tensor(1/(1+np.exp(-self.data)), [self], "sigmoid",True)
        return Tensor(1/(1+np.exp(-self.data)))
    
    # Гиперболический тангенс
    def tanh(self):
        if self.autograd:
            return Tensor(np.tanh(self.data), [self], "tanh",True)
        return Tensor(np.tanh(self.data))

    # Функция активации ReLU (Rectified Linear Unit)
    def relu(self):  
        if self.autograd:
            return Tensor((self.data > 0) * self.data, [self], "relu",True)
        return (self.data > 0) * self.data
    
    # Представление объекта для отладки
    def __repr__(self):
        return str(self.data.__repr__())


#
# SGD - класс для стохастического градиентного спуска
#
class SGD(object):
    # Конструктор оптимизатора SGD
    def __init__(self, weights, learning_rate = 0.01):
        # Список обучаемых весов
        self.weights = weights
        # Скорость обучения
        self.learning_rate = learning_rate

    # Шаг оптимизации - обновление весов
    def step(self):
        for weight in self.weights:
            # Обновление веса: w = w - η * ∇w
            weight.data -= self.learning_rate * weight.grad.data
            # Обнуление градиента после обновления
            weight.grad.data *= 0




# Тестирование функции ReLU

# Создание тестового тензора с включенным автоградом
a = Tensor([[2,3,4],[2,3,5]], autograd=True)
# Применение функции ReLU
a2 = a.relu()
# Вызов обратного распространения с произвольным градиентом
a2.backward(Tensor([4,5,10]))
# Вывод градиента
print(a2.grad)

# Установка seed для воспроизводимости случайных чисел
np.random.seed(0)

# Инициализация весов нейронной сети
weights = [
    Tensor(np.random.randn(3, 3), autograd=True),  # Веса первого слоя
    Tensor(np.random.randn(3, 3), autograd=True),  # Веса второго слоя
    Tensor(np.random.randn(3, 1), autograd=True)   # Веса выходного слоя
]

# Создание оптимизатора SGD
sgd = SGD(weights, 0.01)

# Обучающие данные: (признаки, целевое значение)
train_data = [
    ([1, 4, 5], 20),
    ([1, 5, 5], 25),
    ([1, 3, 10], 30),
    ([1, 4, 8], 32),
    ([1, 5, 8], 40),
    ([1, 5, 9], 45),
    ([1, 6, 8], 48),
    ([1, 7, 7], 49),
    ([1, 5, 10], 50),
    ([1, 6, 9], 54),
    ([1, 7, 8], 56),
    ([1, 8, 8], 64),
    ([1, 7, 10], 70),
    ([1, 8, 9], 72),
    ([1, 9, 9], 81),
    ([2, 5, 5], 50),
    ([2, 4, 7], 56),
    ([2, 5, 7], 70),
    ([2, 6, 7], 84),
    ([2, 4, 11], 88),
    ([2, 5, 9], 90),
    ([2, 6, 8], 96),
    ([2, 5, 10], 100),
    ([3, 3, 3], 27),
    ([3, 3, 4], 36),
    ([3, 3, 5], 45),
    ([3, 4, 7], 84),
    ([3, 5, 4], 60)
]

# Обучение нейронной сети
for epoch in range(1000):
    for inputs, target in train_data:
        # Создание тензора входных данных
        inp = Tensor([inputs], autograd=True)
        # Создание тензора целевых значений
        true_predictions = Tensor([[target]], autograd=True)

        # Прямой проход через нейронную сеть:
        # Умножение на веса -> сигмоида -> умножение на веса -> сигмоида -> умножение на веса
        prediction = inp.dot(weights[0]).sigmoid().dot(weights[1]).sigmoid().dot(weights[2])
        
        # Вычисление ошибки (квадратичная)
        error = (prediction - true_predictions) * (prediction - true_predictions)
        
        # Обратное распространение ошибки
        error.backward()
        # Обновление весов
        sgd.step()

    # Вывод ошибки каждые 20 эпох
    if epoch % 20 == 0:
        print("Epoch", epoch, "error =", error.data)


# Тестирование обученной модели
test_inp = Tensor([[3,5,4]], autograd=False)
test_pred = test_inp.dot(weights[0]).sigmoid().dot(weights[1]).sigmoid().dot(weights[2])
print("Prediction for (3,5,4):", test_pred.data)
