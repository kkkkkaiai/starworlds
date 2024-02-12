import numpy as np
import matplotlib.pyplot as plt

# 定义函数和梯度
def f(theta):
    return theta

def df(theta):
    return np.ones_like(theta)




def test1():
    # 绘制函数曲线和 normal vector
    fig, ax = plt.subplots()
    ax = plt.subplot(211)
    theta_vals = np.linspace(0, 2*np.pi, 100)
    r_vals = f(theta_vals)
    ax.plot(theta_vals, r_vals, label='Function')

    r = f(theta_vals)
    ms = df(theta_vals)
    mn = -1/ms
    thetan = np.arctan(mn)

    rn = 1/np.sqrt(1 + mn**2)

    xn = 1 * np.cos(thetan)
    yn = 1 * np.sin(thetan)

    ax.quiver(theta_vals, r, xn, yn, scale=10, color='g', label='Normal Vector')

    # 添加标签和标题
    ax.set_title('Normal Vector in Polar Coordinates')
    ax.legend()
    ax.axis('equal')

    ax = plt.subplot(212)

    # retrieve the polar coordinate into the cartesian coordinate
    x = r * np.cos(theta_vals)
    y = r * np.sin(theta_vals)

    # for i in range(len(xn)):
    #     xn[i] = thetan[i]*np.sin(theta_vals[i])+r[i]*np.cos(theta_vals[i])
    #     yn[i] = -thetan[i]*np.cos(theta_vals[i])+r[i]*np.sin(theta_vals[i])

    for i in range(len(xn)):
        
        xn[i] = ms[i]*np.cos(theta_vals[i])
        yn[i] = 1 * np.sin(theta_vals[i]+thetan[i])
        print(theta_vals[i]+thetan[i], xn[i], yn[i])


    # plot the function in the cartesian coordinate
    ax.plot(x, y, label='Function')
    ax.quiver(x, y, xn, yn, scale=10, color='g', label='Normal Vector')
    ax.axis('equal')
    # 显示图像
    plt.show()

def test2():
    # 绘制函数曲线和 normal vector
    fig, ax = plt.subplots()
    theta_vals = np.linspace(0, 2*np.pi, 100)
    r_vals = f(theta_vals)


    # y = f(theta) * cos(theta)
    # x = f(theta) * sin(theta)
    # y' = f'(theta) * cos(theta) - f(theta) * sin(theta)
    # x' = f'(theta) * sin(theta) + f(theta) * cos(theta)
    d_theta = df(theta_vals)
    y_dot = d_theta * np.cos(theta_vals) - r_vals * np.sin(theta_vals)
    x_dot = d_theta * np.sin(theta_vals) + r_vals * np.cos(theta_vals)

    x = r_vals * np.cos(theta_vals)
    y = r_vals * np.sin(theta_vals)

    # derive the normal vector
    thetan = np.arctan(y_dot / x_dot)
    x_dot = 0.5 * np.cos(thetan)
    y_dot = -0.5 * np.sin(thetan)

    # judge the direction of the normal vector
    for i in range(len(x_dot)):
        # if the normal vector's direction is same as the direction from the point to the center, reverse the normal vector 
        if (x[i] * x_dot[i] + y[i] * y_dot[i]) < 0:
            x_dot[i] = -x_dot[i]
            y_dot[i] = -y_dot[i]

    # driection

    # normalize the normal vector
    # norm = np.sqrt(x_dot**2 + y_dot**2)
    # x_dot = x_dot / norm
    # y_dot = y_dot / norm

    ax.plot(x, y, label='Function')
    ax.quiver(x, y, x_dot, y_dot, scale=10, color='g', label='Normal Vector')
    plt.show()

test2()
