# 开发时间: 2023/5/15 21:45
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.datasets import cifar10
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
# 加载CIFAR10数据集
file_path=r"D:\PythonCode\pythonProject\data\cifar-10-batches-py\data_batch_1"
f =open()
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# 对数据进行预处理
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, 3072))
X_test = scaler.transform(X_test.reshape(-1, 3072))
# 训练K-NN模型
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
# 在测试集上进行预测
y_pred = knn.predict(X_test)
# 计算分类准确率
accuracy = accuracy_score(y_test, y_pred)
print("分类准确率：", accuracy)