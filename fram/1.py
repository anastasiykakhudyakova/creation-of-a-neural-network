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

#Добавление автоматической дифференциации-это  autograd


"""Задания 
1.Напишите код из параграфа "Тензоры".
Добавление автоматической дифференциации
1.Напишите пример, который проверит правильность кода с автоградиентом.
2.Измените генерацию ID, применив статическую переменную. 
То есть должна быть переменная, которая в начале равна 0, именно она и будет присваиваться в качестве id. 
А после каждого присвоения она будет увеличиваться на 1. Чем этот алгоритм лучше, чем представленный в уроке?"""
import numpy as np

class Tensor(object): #создм каеласс Tensor, наследуем от object
    # Статическая переменная для генерации ID
    _next_id = 0
    
    def __init__(self, data, creators=None, operation_on_creation=None, autograd=False, id=None): # конструктор, принимающий сами данные
#layer_out вычислен из layer_hid и weight_out , значит нам нужно иметь доступ к значениям обеих переменных.
#Поэтому мы добавляем в конструктор класса два параметра:
#creators – те кто создал тензор
#operation_on_creation – операция необходимая для правильного расчета
#градиента при обратном распространении: dot, nm и т.д.
        self.data = np.array(data)# преобразуем данные в массив np
        self.creators = creators
        self.operation_on_creation = operation_on_creation
        self.autograd = autograd# нам нужна информация о всех дочерних элементах тензора, т.е. элементах в создании которых участвовал данный тензор
        self.grad = None # предварительное объявление поля градиента
        # если id пустой, то мы его генерируем
        # Используем статическую переменную для генерации ID
        if id is None:
            id = Tensor._next_id
            Tensor._next_id += 1
        self.id = id
        
        # Словарь для отслеживания дочерних элементов
        self.children = {}
        
        # Сообщаем создателям о себе
# проверяем есть ли для нашего тензора создатели и если есть, то сообщаем им о себе, добавляя в словарь дочерних элементах
        if creators is not None:
            for creator in creators:
                if self.id not in creator.children:
                    creator.children[self.id] = 1
                else:
                    creator.children[self.id] += 1# если мы уже сообщади о себе, то просто увеличиваем количество детей от этого id
    
    def __add__(self, other):# объявим функцию сложения, которая будет возвращать новый тензор, как сумму двух тензоров
# если у складываемых тензоров включен автоградиент, то мы изменяем результат возвращаемый функцией. если автоградиент не включен, то информацию о создателях передавать не нужно.
        if self.autograd and other.autograd:
            return Tensor(self.data + other.data,[self, other],"+", True)#при создании нового тензора мы соответственно должны передавать его создателей и операцию, в результате которой он создался.
        return Tensor(self.data + other.data)
    
    def __sub__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data - other.data,
                         [self, other],
                         "-",
                         True)
        return Tensor(self.data - other.data)
    
    def __mul__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data * other.data,
                         [self, other],
                         "*",
                         True)
        return Tensor(self.data * other.data)
    
    def __matmul__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data @ other.data,
                         [self, other],
                         "@",
                         True)
        return Tensor(self.data @ other.data)
    
    def __str__(self):# функция нужна для удобного вывода в консоль
        return f"Tensor(id={self.id}, data={self.data}, grad={self.grad.data if self.grad else None})"
    
    def __repr__(self):
        return self.__str__()
# реализуем функцию обратного распространения при операции "+". Данная операция характеризуется неизменным градиентом. В случае другой операции градиент необходимо пересчитать
    def backward(self, grad=None, grad_origin=None):# grad-origin - это тензор, созданный благодаря текущему тензору

#Зачем нужна переменная grad_origin (тензор, созданный благодаря текущему тензору)?
#пока мы не получим информацию обо всех его потомках (тензоры 2 и 3 являются grad_origin). Мы должны посчитать градиент элемента только после того как получен градиент от всех его дочерних элементов.
 # добвление автоградиента позволяет выполнять код ниже только в случае, когда автоградиент включен, иначе делать ничего не нужно
        if self.autograd:
            # Если градиент не передан, создаем единичный
            if grad is None:
                grad = Tensor(np.ones_like(self.data))
            
            # Если передан создатель, уменьшаем счетчик дочерних элементов
            if grad_origin is not None:
                if self.children[grad_origin.id] > 0:
                    self.children[grad_origin.id] -= 1
                else:
                    raise Exception("Нет дочернего элемента с таким ID")
            
            # Инициализируем или суммируем градиент
            if self.grad is None:
                self.grad = grad
            else:
                self.grad.data += grad.data# суммируем старое и новое значения градиентов
# теперь мы должны выполнить проверки и произвести вычисления
            
            # Проверяем, все ли градиенты от дочерних элеменов получены
            if self.creators is not None and (self.check_grads_from_children() or grad_origin is None):
                if self.operation_on_creation == "+":
                    # Для сложения градиент передается без изменений
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad, self)
                elif self.operation_on_creation == "-":
                    # Для вычитания первый получает градиент, второй - отрицательный градиент
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(Tensor(-self.grad.data), self)
                elif self.operation_on_creation == "*":
                    # Для умножения: dL/da = dL/dc * b, dL/db = dL/dc * a
                    self.creators[0].backward(Tensor(self.grad.data * self.creators[1].data), self)
                    self.creators[1].backward(Tensor(self.grad.data * self.creators[0].data), self)
                elif self.operation_on_creation == "@":
                    # Для матричного умножения
                    self.creators[0].backward(Tensor(self.grad.data @ self.creators[1].data.T), self)
                    self.creators[1].backward(Tensor(self.creators[0].data.T @ self.grad.data), self)
    
    def check_grads_from_children(self):
        """Проверяет, получены ли градиенты от всех дочерних элементов"""
        for child_id in self.children: # функция возвращает true когда все градиенты детей получены
            if self.children[child_id] != 0:
                return False
        return True


# ========== ЗАДАНИЕ 1: ПРОВЕРКА ПРАВИЛЬНОСТИ КОДА С АВТОГРАДИЕНТОМ ==========

print("=" * 60)
print("ЗАДАНИЕ 1: Проверка правильности кода с автоградиентом")
print("=" * 60)

# Тест 1: Проверка сложения с общим элементом
print("\nТест 1: Сложение с общим элементом (исходный пример)")
a_1 = Tensor([1, 2, 3], autograd=True)
a_2 = Tensor([1, 2, 3], autograd=True)
a_3 = Tensor([1, 2, 3], autograd=True)

a_add_1 = a_1 + a_2
a_add_2 = a_2 + a_3
a_add_3 = a_add_1 + a_add_2

print(f"a_1: {a_1}")
print(f"a_2: {a_2}")
print(f"a_3: {a_3}")
print(f"a_add_1 = a_1 + a_2: {a_add_1}")
print(f"a_add_2 = a_2 + a_3: {a_add_2}")
print(f"a_add_3 = a_add_1 + a_add_2: {a_add_3}")

# Выполняем обратное распространение
a_add_3.backward(Tensor([4, 5, 3]))

print(f"\nПосле backward с градиентом [4, 5, 3]:")
print(f"Градиент a_1: {a_1.grad.data if a_1.grad else 'Нет градиента'}")
print(f"Градиент a_2: {a_2.grad.data if a_2.grad else 'Нет градиента'}")
print(f"Градиент a_3: {a_3.grad.data if a_3.grad else 'Нет градиента'}")

# Проверяем правильность
expected_a2_grad = [8, 10, 6]  # a_2 участвовал дважды: 4+4=8, 5+5=10, 3+3=6
actual_a2_grad = a_2.grad.data if a_2.grad else None
print(f"\nОжидаемый градиент a_2: {expected_a2_grad}")
print(f"Фактический градиент a_2: {actual_a2_grad}")
print(f"Правильно ли работает автоградиент? {np.array_equal(actual_a2_grad, expected_a2_grad)}")

# Тест 2: Проверка умножения
print("\n\nТест 2: Проверка умножения")
b_1 = Tensor([2, 3, 4], autograd=True)
b_2 = Tensor([1, 2, 3], autograd=True)
b_mul = b_1 * b_2

print(f"b_1: {b_1}")
print(f"b_2: {b_2}")
print(f"b_mul = b_1 * b_2: {b_mul}")

b_mul.backward(Tensor([1, 1, 1]))

print(f"\nПосле backward с градиентом [1, 1, 1]:")
print(f"Градиент b_1: {b_1.grad.data if b_1.grad else 'Нет градиента'}")
print(f"Градиент b_2: {b_2.grad.data if b_2.grad else 'Нет градиента'}")

# Проверяем правильность: d(b_mul)/db_1 = b_2, d(b_mul)/db_2 = b_1
expected_b1_grad = [1, 2, 3]  # b_2
expected_b2_grad = [2, 3, 4]  # b_1
print(f"\nОжидаемый градиент b_1: {expected_b1_grad}")
print(f"Фактический градиент b_1: {b_1.grad.data if b_1.grad else 'Нет градиента'}")
print(f"Ожидаемый градиент b_2: {expected_b2_grad}")
print(f"Фактический градиент b_2: {b_2.grad.data if b_2.grad else 'Нет градиента'}")

# Тест 3: Проверка цепочки операций
print("\n\nТест 3: Цепочка операций")
c_1 = Tensor([2, 3], autograd=True)
c_2 = Tensor([4, 5], autograd=True)
c_3 = Tensor([1, 2], autograd=True)

c_sum = c_1 + c_2
c_result = c_sum * c_3

print(f"c_1: {c_1}")
print(f"c_2: {c_2}")
print(f"c_3: {c_3}")
print(f"c_sum = c_1 + c_2: {c_sum}")
print(f"c_result = c_sum * c_3: {c_result}")

c_result.backward(Tensor([1, 1]))

print(f"\nПосле backward с градиентом [1, 1]:")
print(f"Градиент c_1: {c_1.grad.data if c_1.grad else 'Нет градиента'}")
print(f"Градиент c_2: {c_2.grad.data if c_2.grad else 'Нет градиента'}")
print(f"Градиент c_3: {c_3.grad.data if c_3.grad else 'Нет градиента'}")

# Проверяем правильность
# d(c_result)/d(c_1) = d(c_result)/d(c_sum) * d(c_sum)/d(c_1) = c_3 * 1 = c_3
# d(c_result)/d(c_2) = d(c_result)/d(c_sum) * d(c_sum)/d(c_2) = c_3 * 1 = c_3
# d(c_result)/d(c_3) = c_sum
expected_c1_grad = [1, 2]  # c_3
expected_c2_grad = [1, 2]  # c_3
expected_c3_grad = [6, 8]  # c_sum (2+4=6, 3+5=8)

print(f"\nОжидаемый градиент c_1: {expected_c1_grad}")
print(f"Фактический градиент c_1: {c_1.grad.data if c_1.grad else 'Нет градиента'}")
print(f"\nОжидаемый градиент c_2: {expected_c2_grad}")
print(f"Фактический градиент c_2: {c_2.grad.data if c_2.grad else 'Нет градиента'}")
print(f"\nОжидаемый градиент c_3: {expected_c3_grad}")
print(f"Фактический градиент c_3: {c_3.grad.data if c_3.grad else 'Нет градиента'}")

# ========== ЗАДАНИЕ 2: ПРЕИМУЩЕСТВА СТАТИЧЕСКОЙ ПЕРЕМЕННОЙ ДЛЯ ID ==========

print("\n" + "=" * 60)
print("ЗАДАНИЕ 2: Преимущества статической переменной для ID")
print("=" * 60)

# Сбрасываем статическую переменную для демонстрации
Tensor._next_id = 0

# Создаем несколько тензоров
print("\nСоздание тензоров со статическим ID:")
tensors = []
for i in range(5):
    t = Tensor([i], autograd=True)
    tensors.append(t)
    print(f"Тензор {i}: ID = {t.id}")

# Преимущества статической переменной для генерации ID:
print("\n\nПреимущества статической переменной для генерации ID:")
print("1. УНИКАЛЬНОСТЬ: Гарантирует уникальность ID в пределах одного запуска программы")
print("2. ПРЕДСКАЗУЕМОСТЬ: ID назначаются последовательно (0, 1, 2, ...), что упрощает отладку")
print("3. ОТСУТСТВИЕ КОЛЛИЗИЙ: Нет риска генерации одинаковых случайных чисел")
print("4. ПРОСТОТА: Не требуется сложная логика для проверки уникальности")
print("5. ВОСПРОИЗВОДИМОСТЬ: При одинаковых условиях получаются одинаковые ID")

print("\n\nНедостатки случайной генерации ID (из урока):")
print("1. ВОЗМОЖНЫ КОЛЛИЗИИ: Случайные числа могут повторяться, особенно при большом количестве тензоров")
print("2. СЛОЖНОСТЬ ОТЛАДКИ: Случайные ID затрудняют воспроизведение ошибок")
print("3. НЕПРЕДСКАЗУЕМОСТЬ: Невозможно предсказать, какой ID получит тензор")
print("4. ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА: Может потребоваться проверка на уникальность")

# Демонстрация работы с матричным умножением
print("\n\nДополнительная демонстрация: матричное умножение")
print("-" * 40)

# Создаем матрицы
matrix_a = Tensor([[1, 2], [3, 4]], autograd=True)
matrix_b = Tensor([[5, 6], [7, 8]], autograd=True)
matrix_c = Tensor([[2, 0], [0, 2]], autograd=True)

# Выполняем операции: (A @ B) * C
mat_mul = matrix_a @ matrix_b
result = mat_mul * matrix_c

print(f"matrix A:\n{matrix_a.data}")
print(f"\nmatrix B:\n{matrix_b.data}")
print(f"\nmatrix C:\n{matrix_c.data}")
print(f"\nA @ B:\n{mat_mul.data}")
print(f"\n(A @ B) * C:\n{result.data}")

# Выполняем обратное распространение
result.backward(Tensor([[1, 1], [1, 1]]))

print(f"\nГрадиент matrix A:\n{matrix_a.grad.data if matrix_a.grad else 'Нет градиента'}")
print(f"\nГрадиент matrix B:\n{matrix_b.grad.data if matrix_b.grad else 'Нет градиента'}")
print(f"\nГрадиент matrix C:\n{matrix_c.grad.data if matrix_c.grad else 'Нет градиента'}")

