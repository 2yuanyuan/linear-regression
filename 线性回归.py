# 导入所需库
import matplotlib.pyplot as plt
import numpy as np

# 生成数据集
x = np.linspace(-3, 3, 30)
y = 2 * x + 1
x = x + np.random.rand(30)

# 序列变成矩阵
x = [[i] for i in x]
y = [[i] for i in y]
# 测试数据x_
x_ = [[1], [2]]

# 从Scikit-Learn库导入线性模型中的线性回归算法
from sklearn import linear_model

# 训练线性回归模型
model = linear_model.LinearRegression()
model.fit(x, y)

# 进行预测
model.predict(x_)
print("Array({})".format(model.predict(x_)))

# 查看w、b
print("W: {} b: {}".format(model.coef_, model.intercept_))
# 数据集绘图
plt.scatter(x, y)
plt.plot(x, model.predict(x))
plt.show()
