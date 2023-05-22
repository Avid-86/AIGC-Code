from utils.features import prepare_for_training
import numpy as np
class LinearRegression:
    #1.对数据进行预处理操作2.先得到所有的特征个数3.初始化参数矩阵
    def __init__(self,data,labels,polynomial_degree=0,sinusoid_degree=0,normalize_data=True):
        (data_processed,
         features_mean,
         features_deviation)=prepare_for_training(data,polynomial_degree=0,sinusoid_degree=0,normalize_data=True)
        self.data=data_processed
        self.labels=labels#对应的y值
        self.features_mean=features_mean
        self.features_deviation=features_deviation
        self.polynomial_degree=polynomial_degree
        self.sinusoid_degree=sinusoid_degree
        self.normalize_data=normalize_data
        num_features= self.data.shape[1]
        self.theta = np.zeros((num_features,1))
    #训练调用
    def train(self,alpha,num_iterations=500):
        cost_history=self.gradient_descent(alpha,num_iterations)
        return self.theta,cost_history
    #要进行多次梯度下降,一共num_iterations次
    def gradient_descent(self,alpha,num_iterations):
        cost_history = []
        for i in range(num_iterations):
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data,self.labels))
        return cost_history
    #一次梯度下降参数更新计算方法，注意是矩阵运算
    def gradient_step(self,alpha):
        num_examples =self.data.shape[0]#共有多少个数据
        predictions = LinearRegression.hypothesis(self.data,self.theta)#得出预测值
        delta = predictions-self.labels#得出误差值
        theta = self.theta
        theta=theta-alpha*(1/num_examples)*(np.dot(delta.T,self.data)).T
        self.theta=theta
    def cost_function(self,data,labels):
        num_examples = data.shape[0]
        delta = LinearRegression.hypothesis(self.data, self.theta) - labels # 得出预测值
        cost=(1/2)*np.dot(delta.T,delta)
        return cost[0][0]
    #计算出预测值
    @staticmethod
    def hypothesis(data,theta):
        predictions = np.dot(data,theta)
        return predictions
    def get_cost(self,data,labels):
        dataprocessed = prepare_for_training(data,
        self.polynomial_degree,
        self.sinusoid_degree,
        self.normalize_data)[0]
        return self.cost_function(dataprocessed,labels)
    #用训练好的参数模型与预测得到回归的结果
    def predict(self, data):
        dataprocessed = prepare_for_training(data,
         self.polynomial_degree,
         self.sinusoid_degree,
         self.normalize_data)[0]
        predictions = LinearRegression.hypothesis(dataprocessed,self.theta)
        return predictions



