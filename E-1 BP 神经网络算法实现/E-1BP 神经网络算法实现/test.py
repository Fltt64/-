import math
import os
import numpy as np

#激活函数
def Tanh(x):
    return np.tanh(x)

"""" 一维输入的神经网络实现"""""
class NeuralNetwork1D:
    def __init__(self):
        self.input_size = 1#输入层1个神经元
        self.hidden_size = 20#隐藏层20个神经元
        self.output_size = 1#输出层1个神经元

        #初始化输入层——>隐藏层的连接权（v1）
        self.v1 = np.random.uniform(-0.5,0.5,size=(1,self.hidden_size))
        #初始化隐藏层的阈值（u1）
        self.u1 = np.random.uniform(-0.5,0.5,size=(1,self.hidden_size))

        # 初始化隐藏层——>输出层的连接权（w1）
        self.w1 = np.random.uniform(-0.5,0.5,size=(self.hidden_size,1))
        # 初始化输出层的阈值（o1）
        self.o1 = np.random.uniform(-0.5,0.5,size=(1,1))

    #前向传播
    def forward(self, x):
        #隐藏层的输入
        self.a1=x@self.v1
        #隐藏层的输出
        self.b1=Tanh(self.a1-self.u1)

        #输出层的输出
        self.y1=self.b1@self.w1-self.o1
        return self.y1
    #训练
    def train(self):
        # 生成随机训练数据
        sample_num = 500
        x_train = np.linspace(0, 2 * math.pi, sample_num).reshape(-1, 1)
        y_train = np.sin(x_train)
        # 学习率,迭代次数
        N = 0.05
        Epochs = 100000
        for epoch in range(Epochs):
            y_pred = self.forward(x_train)#预测值
            dz2 = 2 * (y_pred - y_train)/sample_num

            # 输出层参数梯度
            dw1 = self.b1.T @ dz2  # w1的梯度
            do1 = -np.sum(dz2, axis=0, keepdims=True)  # o1的梯度（减阈值，多一个负号）

            #隐藏层梯度
            db1 = dz2 @ self.w1.T  # 损失对隐藏层激活值的偏导
            da1 = db1 * (1 - self.b1 ** 2)  # 乘Tanh导数，得到对净输入a1的偏导

            dv1 = x_train.T @ da1  # v1的梯度
            du1 = -np.sum(da1, axis=0, keepdims=True)  # u1的梯度（同理，减阈值加负号）

            # 梯度下降更新所有参数
            self.v1 -= N * dv1
            self.u1 -= N * du1
            self.w1 -= N * dw1
            self.o1 -= N * do1

    def predict(self, input_x: float) -> float:
        x = np.array([[input_x]])
        result = self.forward(x)
        return float(result[0, 0])
#===============================================================================
#===============================================================================
#===============================================================================
""" 二维输入的神经网络实现"""
class NeuralNetwork2D:
    def __init__(self):
        self.input_size = 2  # 输入层2个神经元
        self.hidden_size = 40  # 隐藏层20个神经元
        self.output_size = 1  # 输出层1个神经元

        # 初始化输入层——>隐藏层的连接权（v1）
        self.v1 = np.random.uniform(-0.5, 0.5, size=(self.input_size, self.hidden_size))
        # 初始化隐藏层的阈值（u1）
        self.u1 = np.random.uniform(-0.5, 0.5, size=(1, self.hidden_size))

        # 初始化隐藏层——>输出层的连接权（w1）
        self.w1 = np.random.uniform(-0.5, 0.5, size=(self.hidden_size, self.output_size))
        # 初始化输出层的阈值（o1）
        self.o1 = np.random.uniform(-0.5, 0.5, size=(1, self.output_size))

    def forward(self, x):
        self.a1=x@self.v1
        self.b1=Tanh(self.a1-self.u1)
        self.y1 = self.b1 @ self.w1 - self.o1
        return self.y1

    def train(self):
        x1 = np.linspace(0, 2 * math.pi, 30)
        x2 = np.linspace(0, 2 * math.pi, 30)
        X1, X2 = np.meshgrid(x1, x2)
        x_train = np.column_stack([X1.ravel(), X2.ravel()])  # (2500, 2)
        y_train = (np.sin(x_train[:, 0]) * np.cos(x_train[:, 1])).reshape(-1, 1)
        sample_num = x_train.shape[0]  # 用实际样本数
        # 学习率,迭代次数
        N = 0.05
        Epochs = 100000
        #初始化动量
        mv1 = np.zeros_like(self.v1)
        mu1 = np.zeros_like(self.u1)
        mw1 = np.zeros_like(self.w1)
        mo1 = np.zeros_like(self.o1)
        beta = 0.9

        for epoch in range(Epochs):
            y_pred = self.forward(x_train)
            dz2 = 2 * (y_pred - y_train) / sample_num

            # 输出层参数梯度
            dw1 = self.b1.T @ dz2  # w1的梯度
            do1 = -np.sum(dz2, axis=0, keepdims=True)  # o1的梯度（减阈值，多一个负号）

            # 隐藏层梯度
            db1 = dz2 @ self.w1.T  # 损失对隐藏层激活值的偏导
            da1 = db1 * (1 - self.b1 ** 2)  # 乘Tanh导数，得到对净输入a1的偏导

            dv1 = x_train.T @ da1  # v1的梯度
            du1 = -np.sum(da1, axis=0, keepdims=True)  # u1的梯度（同理，减阈值加负号）

            # 梯度下降更新所有参数
            mv1 = beta * mv1+ N * dv1
            self.v1 -= mv1

            mu1 = beta * mu1 +N * du1
            self.u1 -= mu1

            mw1 = beta * mw1 + N*dw1
            self.w1 -= mw1

            mo1 = beta * mo1 + N* do1
            self.o1 -= mo1

    def predict(self, input_x1: float, input_x2: float) -> float:
        x = np.array([[input_x1, input_x2]])
        result = self.forward(x)
        return float(result[0, 0])

# 不要改动此类
class Test:
    def __init__(self, num: int):
        self.num = num
        if num == 0:
            self.net1 = NeuralNetwork1D()
            self.net1.train()
        else:
            self.net2 = NeuralNetwork2D()
            self.net2.train()

    def output_y(self, *args) -> float:
        if self.num == 0:
            return self.net1.predict(args[0])
        else:
            return self.net2.predict(args[0], args[1])

    def testbench(self):
        sum_error = 0.0

        if self.num == 0:
            total = 500
            for i in range(total):
                x = 1.0 * i / total * 2 * math.pi
                y = self.output_y(x)
                sum_error += abs(math.sin(x) - y)
            average_error = sum_error / total
        else:
            total = 20
            for i in range(total):
                for j in range(total):
                    x1 = 1.0 * i / total * 2 * math.pi
                    x2 = 1.0 * j / total * 2 * math.pi
                    y = self.output_y(x1, x2)
                    true_y = math.sin(x1) * math.cos(x2)
                    sum_error += abs(true_y - y)
            average_error = sum_error / (total * total)

        label = "The 2D is " if self.num else "The 1D is "
        if average_error <= 1e-2:
            print(f"{label}Success! Average: {average_error}")
        else:
            print(f"{label}Failure! Average: {average_error}")


if __name__ == "__main__":
    num = 0  # 参数为0或1，参数为0的时候输入一维度，参数为1的时候输入二维
    t = Test(num)
    t.testbench()
