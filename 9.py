#Измените веса нейросети таким образом, 
#чтобы выходные данные для первого и второго нейрона стали равными. 
#Используйте метод проб и ошибок. Входные значения менять нельзя.
#Выполните предыдущее задание, но с помощью цикла. После цикла выведите получившиеся веса.
def network(inp, weight):
    # Вычисляем выходы нейронов
    predict = [sum([inp[j] * weight[i][j] for j in range(len(inp))]) for i in range(len(weight))] #создаем список выходных значений для каждого нейрона
    return predict

inp = [50, 165, 45]
weights_2 = [0.3, 0.1, 0.7]  # Фиксированные веса второго нейрона

# Простой перебор с разумным диапазоном
for w1 in range(-100, 100):  # от -10.0 до 10.0
    for w2 in range(-100, 100):
        for w3 in range(-100, 100):
            weights_1 = [w1 * 0.1, w2 * 0.1, w3 * 0.1]  # Преобразуем в десятичные
            weights = [weights_1, weights_2]
            predict = network(inp, weights)
            
            # Проверяем равенство выходов (с небольшой погрешностью)
            if abs(predict[0] - predict[1]) < 0.001:
                print(f"Выходы: {predict}")
                print(f"Веса первого нейрона: {weights_1}")
                break
        else:
            continue
        break
    else:
        continue
    break
