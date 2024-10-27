import numpy as np

def evaluation_map(prediction, truth):
    # prediction 为输入的预测答案 应该是一个二维数组，形状为(the number of the question, 3)
    # truth 为输入的真实答案 应该是一个二维数组，形状为(the number of the question, 1)

    prediction = np.array(prediction)
    truth = np.array(truth)

    mix = np.concatenate((prediction, truth), axis=1) # 合并为(the number of the question, 4)的数组，最后一列是答案

    #mix.shape # 测试

    for x in mix:
        for y in range(0, 3):
            if(mix(x, y) == mix(x, 3)): # 遍历判断每一个值是否相等
                tru = tru + 1
        p_k = p_k + tru / 3 # sum of the right answer / 3 = P(k)
        tru = 0
    U = mix.shape[0] # sum / the number of the question
    return p_k / U







