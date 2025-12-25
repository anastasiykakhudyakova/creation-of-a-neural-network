# Импорт библиотеки NumPy для работы с массивами
import numpy as np

# Определение основного класса Tensor для автоматического дифференцирования
class Tensor(object):
    
    # Счетчик для присвоения уникальных ID каждому тензору
    id_count = 0

    # Основной конструктор класса Tensor
    def __init__(self, data, creators=None, operation_on_creation=None, autograd=False, id=None):
        # Вызов метода инициализации с переданными параметрами
        self.init(data, creators, operation_on_creation, autograd, id)

    # Метод инициализации тензора
    def init(self, data, creators=None, operation_on_creation=None, autograd=False, id=None):
        # Преобразование входных данных в массив NumPy
        self.data = np.array(data)
        # Родительские тензоры, от которых произошел данный тензор
        self.creators = creators
        # Операция, которая создала этот тензор (для обратного распространения)
        self.operation_on_creation = operation_on_creation
        # Градиент тензора (производная потерь по данному тензору)
        self.grad = None
        # Флаг, указывающий, нужно ли вычислять градиенты для этого тензора
        self.autograd = autograd
        # Словарь для отслеживания количества зависимых тензоров (детей)
        self.children = {}
        
        # Если ID не задан, генерируем новый уникальный ID
        if id is None:
            self.__class__.id_count += 1
            self.id = self.__class__.id_count
            
        # Если есть родители, регистрируем себя в их детях
        if self.creators is not None:
            for creator in creators:
                if self.id not in creator.children:
                    creator.children[self.id] = 1
                else:
                    creator.children[self.id] += 1

    # Метод обратного распространения ошибки
    def backward(self, grad=None, grad_origin=None):
        # Проверяем, нужно ли вычислять градиенты для этого тензора
        if self.autograd:
            # Если градиент не передан, инициализируем его единицами
            if grad is None:    
                grad = Tensor(np.ones_like(self.data))
            # Если указан источник градиента (тензор-ребенок)
            if grad_origin is not None:
                # Уменьшаем счетчик ожидаемых градиентов от этого ребенка
                if self.children[grad_origin.id] > 0:
                    self.children[grad_origin.id] -= 1
                            
        # Инициализация или накопление градиента
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad
        # Рекурсивный вызов backward для родителей, если все градиенты от детей получены
        if self.creators is not None and (self.check_grads_from_children() or grad_origin is None):
                # Обратное распространение для операции сложения
                if self.operation_on_creation == "+":
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad, self)
                    
                # Обратное распространение для унарного минуса
                elif self.operation_on_creation == "-1":
                    self.creators[0].backward(self.grad.__neg__(), self)
                    
                # Обратное распространение для вычитания
                elif self.operation_on_creation == "-":
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad.__neg__(), self)
                    
                # Обратное распространение для умножения
                elif self.operation_on_creation == "*":
                    new_grad = self.grad * self.creators[1]
                    self.creators[0].backward(new_grad, self)
                    new_grad = self.grad * self.creators[0]
                    self.creators[1].backward(new_grad, self)
                    
                # Обратное распространение для суммы по оси
                elif "sum" in self.operation_on_creation:
                    axis = int(self.operation_on_creation.split("_")[1])
                    self.creators[0].backward(self.grad.expand(axis, self.creators[0].data.shape[axis]), self)
                    
                # Обратное распространение для расширения по оси
                elif "expand" in self.operation_on_creation:
                    axis = int(self.operation_on_creation.split("_")[1])
                    self.creators[0].backward(self.grad.sum(axis), self)
                    
                # Обратное распространение для транспонирования
                elif self.operation_on_creation == "transpose":
                    self.creators[0].backward(self.grad.transpose(), self)
                    
                # Обратное распространение для матричного умножения
                elif self.operation_on_creation == "dot":
                    temp = self.grad.dot(self.creators[1].transpose())
                    self.creators[0].backward(temp, self)
                    temp = self.creators[0].transpose().dot(self.grad)
                    self.creators[1].backward(temp, self)

                # Обратное распространение для сигмоиды
                elif self.operation_on_creation == "sigmoid":
                    temp = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad * (self*(temp - self)), self)

                # Обратное распространение для гиперболического тангенса
                elif self.operation_on_creation == "tanh":
                    temp = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad * (temp - self * self), self)

                # Обратное распространение для ReLU
                elif self.operation_on_creation == "relu":
                    temp = self.grad * Tensor((self.creators[0].data > 0) * 1.0)
                    self.creators[0].backward(temp, self)
                
                # Обратное распространение для softmax
                elif self.operation_on_creation == "softmax":
                    self.creators[0].backward(Tensor(self.grad.data), self)
                
                # Обратное распространение для возведения в степень
                elif self.operation_on_creation == "pow":
                    num = self.creators[0]
                    power = self.creators[1].data
                    new_grad = self.grad * Tensor(power * (num.data ** (power - 1)))
                    num.backward(new_grad, self)
                                    
    # Проверка, получены ли все градиенты от детей
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

    # Перегрузка оператора сложения
    def __add__(self, other):
        return self.add(other)
    
    # Строковое представление тензора
    def __str__(self):
        return str(self.data)

    # Перегрузка оператора вычитания
    def __sub__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data - other.data, [self,other], "-", True)
        return Tensor(self.data - other.data)

    # Перегрузка унарного минуса
    def __neg__(self):
        if self.autograd:
            return Tensor(self.data * (-1), [self], "-1", True)
        return Tensor(self.data * (-1))

    # Перегрузка оператора умножения
    def __mul__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data * other.data, [self, other], "*", True)
        return Tensor(self.data * other.data)

    # Суммирование по оси
    def sum(self, axis):
        if self.autograd:
            return Tensor(self.data.sum(axis), [self], "sum_" + str(axis), True)
        return Tensor(self.data.sum(axis))
    
    # Расширение тензора по оси
    def expand(self, axis, count_copies):
        transpose = list(range(0, len(self.data.shape)))
        transpose.insert(axis, len(self.data.shape))
        expand_shape = list(self.data.shape) + [count_copies]
        expand_data = (self.data.repeat(count_copies).reshape(expand_shape))
        expand_data = expand_data.transpose(transpose)
        if self.autograd:
            return Tensor(expand_data, [self], "expand_" + str(axis), True)
        return Tensor(expand_data)

    # Матричное умножение
    def dot(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data.dot(other.data), [self, other], "dot", True)
        return Tensor(self.data.dot(other.data))

    # Транспонирование
    def transpose(self):
        if self.autograd:
            return Tensor(self.data.transpose(), [self], "transpose", True)
        return Tensor(self.data.transpose())

    # Сигмоидальная активация
    def sigmoid(self):
        if self.autograd:
            return Tensor(1/(1+np.exp(-self.data)), [self], "sigmoid",True)
        return Tensor(1/(1+np.exp(-self.data)))
    
    # Гиперболический тангенс
    def tanh(self):
        if self.autograd:
            return Tensor(np.tanh(self.data), [self], "tanh",True)
        return Tensor(np.tanh(self.data))

    # Функция активации ReLU
    def relu(self):  
        if self.autograd:
            return Tensor((self.data > 0) * self.data, [self], "relu",True)
        return (self.data > 0) * self.data
    
    # Представление для отладки
    def __repr__(self):
        return str(self.data.__repr__())
    
    # Функция softmax
    def softmax(self):
        exp = np.exp(self.data)
        exp = exp / np.sum(exp, axis=1, keepdims=True)
        if self.autograd:
            return Tensor(exp, [self], "softmax", True)
        return Tensor(exp)
    
    # Возведение в степень
    def __pow__(self, power):
        if self.autograd:
            power_tensor = Tensor(power, autograd=False)
            return Tensor(self.data ** power, [self, power_tensor], "pow", True)
        return Tensor(self.data ** power)


# Класс для стохастического градиентного спуска
class SGD(object):
    def __init__(self, weights, learning_rate = 0.01):
        # Веса модели для обновления
        self.weights = weights
        # Скорость обучения
        self.learning_rate = learning_rate

    # Шаг оптимизации
    def step(self):
        for weight in self.weights:
            # Обновление весов по градиенту
            weight.data -= self.learning_rate * weight.grad.data
            # Обнуление градиентов
            weight.grad.data *=0


# Базовый класс для слоев нейронной сети
class Layer(object):
    def __init__(self):
        # Список обучаемых параметров слоя
        self.parameters = []

    # Получение параметров слоя
    def get_parameters(self):
        return self.parameters


# Линейный (полносвязный) слой
class Linear(Layer):
    def __init__(self, input_count, output_count):
        # Вызов конструктора родительского класса
        super().__init__()

        # Инициализация весов методом He
        weight = np.random.randn(input_count, output_count) * np.sqrt(2.0/input_count)

        # Создание тензоров для весов и смещений
        self.weight = Tensor(weight, autograd=True)
        self.bias = Tensor(np.zeros(output_count), autograd=True)
        # Добавление параметров в список
        self.parameters.append(self.weight)
        self.parameters.append(self.bias)

    # Прямой проход через слой
    def forward(self, inp):
        return inp.dot(self.weight) + self.bias.expand(0, len(inp.data))


# Последовательная модель из нескольких слоев
class Sequential(Layer):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
    
    # Добавление слоя в модель
    def add(self, layer):
        self.layers.append(layer)

    # Прямой проход через все слои
    def forward(self, inp):
        for layer in self.layers:
            inp = layer.forward(inp)
        return inp
    
    # Получение всех параметров модели
    def get_parameters(self):
        params = []
        for layer in self.layers:
            params += layer.get_parameters()
        return params

# Слой сигмоидальной активации
class Sigmoid(Layer):
    def forward(self, inp):
        return inp.sigmoid()

# Слой гиперболического тангенса
class Tanh(Layer):
    def forward(self, inp):
        return inp.tanh()

# Функция потерь MSE (Mean Squared Error)
class MSELoss(Layer):
    def forward(self, prediction, true_prediction):
        diff = prediction - true_prediction
        return (diff * diff).sum(0) * Tensor(1.0 / prediction.data.shape[0], autograd=True)

# Слой softmax
class Softmax(Layer):
    def forward(self, inp):
        return inp.softmax()

# Функция потерь RMSE (Root Mean Squared Error)
class RMSELoss(Layer):
    def forward(self, prediction, true_prediction):
        return MSELoss().forward(prediction, true_prediction) ** 0.5


# Пример использования для бинарной классификации по весу и возрасту
x = Tensor([
    [80, 25],  # мужчина 80 кг, 25 лет
    [90, 30],  # мужчина 90 кг, 30 лет
    [85, 35],  # мужчина 85 кг, 35 лет
    [45, 22],  # женщина 45 кг, 22 года
    [55, 28],  # женщина 55 кг, 28 лет
    [50, 35],  # женщина 50 кг, 35 лет
], autograd=True)

y = Tensor([
    [1, 0],  # мужчина (класс 0)
    [1, 0],  # мужчина (класс 0)
    [1, 0],  # мужчина (класс 0)
    [0, 1],  # женщина (класс 1)
    [0, 1],  # женщина (класс 1)
    [0, 1],  # женщина (класс 1)
], autograd=True)

# Создание модели: 2 входных нейрона -> 8 скрытых -> 2 выходных
model = Sequential([
    Linear(2, 8),  # Входной слой
    Tanh(),        # Активация
    Linear(8, 2),  # Выходной слой
    Sigmoid()      # Активация (вероятности)
])

# Функция потерь и оптимизатор
loss = MSELoss()
sgd = SGD(model.get_parameters(), learning_rate=0.01)

# Обучение модели
for epoch in range(10000):
    # Прямой проход
    preds = model.forward(x)
    # Вычисление ошибки
    error = loss.forward(preds, y)
    
    # Обратное распространение
    error.backward(Tensor(np.ones_like(error.data)))
    # Обновление весов
    sgd.step()
    
    # Вывод ошибки каждые 500 эпох
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss {error}")

# Функция для предсказания
def predict(weight, age):
    out = model.forward(Tensor([[weight, age]]))
    return out.data

# Тестирование модели
print("Мужчина 80 кг, 25 лет:", predict(80, 25))
print("Женщина 55 кг, 30 лет:", predict(55, 30))
