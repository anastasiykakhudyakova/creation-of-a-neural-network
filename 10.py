#Измените веса нейросети так, 
#чтобы предсказанные значения для второго слоя (prediction_h) стали больше 5. 
#Напишите код, который это сделает. 
#Выведите получившиеся веса. Само собой, входные данные менять нельзя.
#Добавление скрытого слоя,будем рассматривать сеть с двумя входами, выходами и двумя нейронами в скрытом слое.
#Здесь весовые коэффициенты в совокупности являются массивом матриц. Нейронные слои представляют векторы.
#Т.е. входной слой нейронов является входом для скрытого слоя. А скрытый слой явлется входным для слоя выходных нейронов.
#Скрытый слой — это внутренний слой нейронов в нейронной сети, который находится между входным и выходным слоями. Он называется "скрытым", потому что:
#Мы видим только входные данные и конечный результат
#Внутренние вычисления этого слоя не видны напрямую пользователю
def network_hidden(inp, weight_i):
    predict = [0] * len(weight_i)
    for i in range(len(weight_i)):
        ws = 0
        for j in range(len(inp)):
            ws += inp[j] * weight_i[i][j]
        predict[i] = ws
    return predict


def network(inp, weight):
    predict_hidden = network_hidden(inp, weight[0])
    predict = [0] * len(weight[1])
    for i in range(len(weight[1])):
        ws = 0
        for j in range(len(predict_hidden)):
            ws += predict_hidden[j] * weight[1][i][j]
        predict[i] = ws
    return predict


inp = [9, 9]

weight_h_1 = [0.4, 0.1] #весовые коэффициенты для первого нейрона скрытого слоя
weight_h_2 = [0.3, 0.2] #весовые коэффициенты для второго нейрона скрытого слоя
weight_out_1 = [0.4, 0.1] #весовые коэффициенты для связи нейронов скрытого слоя и выходного
weight_out_2 = [0.3, 0.1] #весовые коэффициенты для связи нейронов скрытого слоя и выходного
weights_h = [weight_h_1, weight_h_2]
weights_out = [weight_out_1, weight_out_2]
weights = [weights_h, weights_out]

print(weights)
predict_h = network_hidden(inp, weights[0])
print(predict_h)

COUNT = 20
STEP = 0.099
for i in range(COUNT):
    for j in range(COUNT):
        for k in range(COUNT):
            for m in range(COUNT):
                new_weight_h_1 = [weight_h_1[0] + i * STEP, weight_h_1[1] + j * STEP] 
                new_weight_h_2 = [weight_h_2[0] + k * STEP, weight_h_2[1] + m * STEP] 
                
                new_weights_h = [new_weight_h_1, new_weight_h_2]
                new_weights_out = [weight_out_1, weight_out_2]
                new_weights = [new_weights_h, new_weights_out]
                
                predict_h = network_hidden(inp, new_weights[0])
                if predict_h[0] >= 5 and predict_h[1] >= 5:
                    weights = new_weights
                    i = COUNT
                    j = COUNT
                    k = COUNT
                    m = COUNT
            
print(weights)
predict_h = network_hidden(inp, weights[0])
print(predict_h)
