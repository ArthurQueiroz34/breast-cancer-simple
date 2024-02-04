import numpy as np

def stepFunction(sum):
    if(sum >= 1):
        return 1
    return 0

def sigmoidFunction(sum):
    return 1 / (1 + np.exp(-sum))

def tahnFunction(sum):
    return(np.exp(sum)) - np.exp(-sum) / np.exp(sum) + np.exp(-sum)

def reluFunction(sum):
    if sum >= 0:
        return sum
    return 0

def linearFunction(sum):
    return sum

def softmaxFunction(x):
    ex = np.exp(x)
    return ex / ex.sum()

test = stepFunction(-1)
test = sigmoidFunction(-0.358)
test = tahnFunction(-0.358)
test = reluFunction(0.358)
test = linearFunction(-0.358)
values = [7.0, 2.0, 1.3]
print(softmaxFunction(values))