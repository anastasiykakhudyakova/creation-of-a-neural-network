#Измените веса нейросети таким образом, 
#чтобы выходные данные для первого и второго нейрона стали равными. 
#Используйте метод проб и ошибок. Входные значения менять нельзя.
def network(inp, weight):
    predict = [0] * len(weight)
    for i in range(len(weight)):
        predict[i] = sum([inp[j] * weight[i][j] for j in range(len(inp))])
    return predict

inp = [50, 165, 45]
weights_1 = [0.3, 0.1, 0.7]
weights_2 = [0.3, 0.1, 0.7] # веса одинаковые, поэтому выходы нейронов будут равны,каждый список содержит по 3 веса (по одному на каждый вход)
weights = [weights_1, weights_2]

print(network(inp, weights))
print(weights)
