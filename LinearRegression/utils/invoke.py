# 开发时间: 2023/5/14 21:32
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from main import LinearRegression
data = pd.read_csv('../data/world-happiness-report-2017.csv')
#得到训练和测试数据
train_data = data.sample(frac=0.8)#80%的内容都是训练集
test_data= data.drop(train_data.index)#除去训练集得到的数据给test_data
input_param_name = 'Economy..GDP.per.Capita.'
output_param_name = 'Happiness.Score'
x_train = train_data[[input_param_name]].values
y_train = train_data[[output_param_name]].values
x_test=test_data[input_param_name].values
y_test=test_data[output_param_name].values
plt.scatter(x_train,y_train,label="train data")
plt.scatter(x_test,y_test,label="test data")
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.legend()
plt.show()
num_iterations = 500
learning_rate = 0.01
linear_regression = LinearRegression(x_train,y_train)
(theta,cost_history)= linear_regression.train(learning_rate,num_iterations)
print("开始时的损失：",cost_history[0])
print("最后的损失：",cost_history[-1])
plt.plot(range(num_iterations),cost_history)
plt.xlabel('iter')
plt.ylabel('cost')
plt.title("cost values")
plt.show()
predictions_num = 100
x_predictions = np.linspace(x_train.min(),x_train.max(),predictions_num).reshape(predictions_num,1)
y_predictions = linear_regression.predict(x_predictions)

plt.scatter(x_train,y_train,label="train data")
plt.scatter(x_test,y_test,label="test data")
plt.plot(x_predictions,y_predictions,'r',label = 'predictions')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.legend()
plt.show()




