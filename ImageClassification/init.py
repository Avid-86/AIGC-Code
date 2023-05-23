import pickle
import numpy as np
import matplotlib.pyplot as plt

f = open(r'D:\PythonCode\pythonProject\data\cifar-10-batches-py\data_batch_1','rb')
c = pickle.load(f,encoding='bytes')
data = c[b'data']
tar = np.array(c[b'labels'])

data = data.reshape([-1,3,32,32])
data = np.transpose(data,[0,2,3,1])

def f(i):
    plt.imshow(data[i])


print()