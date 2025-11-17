#задание 2 создайте список входных данных (например, inputs = [150, 160, 170, 180, 190]) и 
# используйте цикл for для вычисления выходных данных нейросети для каждого значения в списке.
# распечатайте выходные данные для каждого входного значения.
def neuralNetwork(inp, weight):
    prediction = inp * weight
    return prediction

inputs = [150, 160, 170, 180, 190]
weight = 0.3

print("Результаты для разных входных данных:")
for inp in inputs:
    result = neuralNetwork(inp, weight)
    print(f"Вход: {inp} -> Выход: {result}")
