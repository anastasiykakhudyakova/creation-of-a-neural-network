#3 задание  модифицируйте функцию neural_network так, чтобы она принимала два входных параметра: inp и bias.
#  Результат будет задан как inp * weight + bias. Запустите функцию с новыми значениями inp, weight и bias. 
# Как изменится выходная переменная? Почему?
def neuralNetwork(inp, weight, bias):
    prediction = inp * weight + bias
    return prediction
out_1  = neuralNetwork(150, 0.3, 0)# 45
out_2 = neuralNetwork(150, 0.3, 10)# 55.0
out_3 = neuralNetwork(150, 0.3, -5)# 40
out_4 = neuralNetwork(150, 0.3, 20)# 65
print(out_1)     
print(out_2)
print(out_3)
print(out_4)
#Bias добавляет смещение к результату, позволяя сети выдавать ненулевой выход даже при нулевом входе
# положительный bias увеличивает выход, отрицательный - уменьшает
# это позволяет смещать "порог активации" нейрона

