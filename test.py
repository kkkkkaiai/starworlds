import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev

# 假设你的x和y数据如下
xdata = np.array([-10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
ydata = np.array([1.2, 4.2, 6.7, 8.3, 10.6, 11.7, 13.5, 14.5, 15.7, 16.1, 16.6, 16.0, 15.4, 14.4, 14.2, 12.7, 10.3, 8.6, 6.1, 3.9, 2.1])

# 使用splrep函数计算样条曲线的表示
tck, u = splrep(xdata, ydata, k=3, s=0)

# 在新的点上评估样条曲线
u_new = np.linspace(0, 1, num=100, endpoint=True)
y_new = splev(u_new, tck)

# 绘制原始数据和拟合的样条曲线
plt.plot(xdata, ydata, 'o', label='原始数据')
plt.plot(xdata, ydata, 'b--', alpha=0.3)
plt.plot(y_new, label='样条曲线拟合')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('样条曲线拟合')
plt.show()