# fit the starshaped function with point set

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from obstacles import StarshapedPolygon
from utils import generate_star_polygon
from test_bezier_fit import bezier_fit


# generate starshaped points
avg_radius = 2
xlim = [-2*avg_radius, 2*avg_radius]
ylim = xlim
starshaped_edge = generate_star_polygon([0, 0], avg_radius, irregularity=0.3, spikiness=0.5, num_vertices=10)
pol = StarshapedPolygon(starshaped_edge, xr=np.array([0, 0]))

starshaped_points = []

print(pol.xr())

for i in np.linspace(0, 2 * np.pi, 150):
    x = pol.xr() + 10*np.array([np.cos(i), np.sin(i)])
    b = pol.boundary_mapping(x)
    starshaped_points.append(b)

# use periodic function to fit the starshaped points
def periodic_func(x, a, b, c, d, e, bias):
    return a * np.sin(b * x + c) + d * np.cos(e * x + c) + bias

# multi periodic function
n = 5
# 
def multi_periodic_func_5(x, bias, a1, b1, d1, e1, a2, b2, d2, e2, a3, b3,d3, e3, a4, b4,d4, e4, a5, b5,d5, e5):
    return a1 * np.sin(b1 * x ) + d1 * np.cos(e1 * x) + a2 * np.sin(b2 * x) + d2 * np.cos(e2 * x) + \
           a3 * np.sin(b3 * x ) + d3 * np.cos(e3 * x) + a4 * np.sin(b4 * x) + d4 * np.cos(e4 * x) + \
           a5 * np.sin(b5 * x ) + d5 * np.cos(e5 * x) + bias

def multi_periodic_func_5_(x, bias, a1, b1, d1, e1, a2, b2, d2, e2, a3, b3,d3, e3, a4, b4,d4, e4, a5, b5, e5, d5):
    return a1 * np.sin(b1 * x ) + d1 * np.cos(e1 * x) + a2 * np.sin(b2 * x) + d2 * np.cos(e2 * x) + \
           a3 * np.sin(b3 * x ) + d3 * np.cos(e3 * x) + a4 * np.sin(b4 * x) + d4 * np.cos(e4 * x) + \
           a5 * np.sin(b5 * x ) + d5 * np.cos(e5 * x) + bias


def multi_periodic_func_5_rbias(x, bias, a1, b1, c1, d1, e1, a2, b2,c2, d2, e2, a3, b3,d3,c3, e3, a4, b4,d4, c4,e4,a5,b5,c5,d5,e5):
    return a1 * np.sin(b1 * x + c1) + d1 * np.cos(e1 * x+c1) + a2 * np.sin(b2 * x+c2) + d2 * np.cos(e2 * x+c2) + \
           a3 * np.sin(b3 * x +c3) + d3 * np.cos(e3 * x+c3) + a4 * np.sin(b4 * x+c4) + d4 * np.cos(e4 * x+c4) + \
           a5 * np.sin(b5 * x +c5) + d5 * np.cos(e5 * x+c5) + bias

def multi_periodic_func_7(x, bias, a1, b1, d1, e1, a2, b2, d2, e2, a3, b3,d3, e3, a4, b4,d4, e4, a5, b5,d5, e5,
                            a6, b6,d6, e6, a7, b7,d7, e7):
    return a1 * np.sin(b1 * x ) + d1 * np.cos(e1 * x) + a2 * np.sin(b2 * x) + d2 * np.cos(e2 * x) + \
              a3 * np.sin(b3 * x ) + d3 * np.cos(e3 * x) + a4 * np.sin(b4 * x) + d4 * np.cos(e4 * x) + \
                a5 * np.sin(b5 * x ) + d5 * np.cos(e5 * x) + a6 * np.sin(b6 * x) + d6 * np.cos(e6 * x) + \
                    a7 * np.sin(b7 * x ) + d7 * np.cos(e7 * x) + bias

# use bezier curve to fit the starshaped points
def bezier_curve_func(x, a, b, c, d, e):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e

# use multi 5 bezier curve to fit the starshaped points
def multi_bezier_curve_func(x, bias, a, b, c, d, e, a1, b1, c1, d1, e1, a2, b2, c2, d2, e2, a3, b3, c3, d3, e3,
                            a4, b4, c4, d4, e4, a5, b5, c5, d5, e5):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e + a1 * x**4 + b1 * x**3 + c1 * x**2 + d1 * x + e1 + \
              a2 * x**4 + b2 * x**3 + c2 * x**2 + d2 * x + e2 + a3 * x**4 + b3 * x**3 + c3 * x**2 + d3 * x + e3 + \
                a4 * x**4 + b4 * x**3 + c4 * x**2 + d4 * x + e4 + a5 * x**4 + b5 * x**3 + c5 * x**2 + d5 * x + e5 \
                    + bias


# use polynomial to fit the starshaped points
def polynomial_func(x, a, b, c, d, e, f, g, h, i, j, k, l):
    return a * x**11 + b * x**10 + c * x**9 + d * x**8 + e * x**7 + f * x**6 + g * x**5 + h * x**4 + i * x**3 + j * x**2 + k * x + l

def polynomic_func_9(x, a, b, c, d, e, f, g, h, i, j):
    return a * x**9 + b * x**8 + c * x**7 + d * x**6 + e * x**5 + f * x**4 + g * x**3 + h * x**2 + i * x + j

# fit the starshaped points
# convert the starshaped points to polar coordinate
starshaped_points = np.array(starshaped_points)
starshaped_center = np.array([0, 0])
starshaped_polar = np.array([np.linalg.norm(starshaped_points[i] - starshaped_center) for i in range(len(starshaped_points))])
# print(starshaped_polar)
# exit()

# gaussian smooth
from scipy.ndimage import gaussian_filter1d
starshaped_polar = gaussian_filter1d(starshaped_polar, sigma=3)


x = np.linspace(0, 2 * np.pi, len(starshaped_polar))
y = starshaped_polar.flatten()

print(x, y)
# calculate computation time
import time
start = time.time()

# segment data to 5 parts

# x1 = x[0:35]
# y1 = y[0:35]
# x2 = x[25:65]
# y2 = y[25:65]
# x3 = x[55:95]
# y3 = y[55:95]
# x4 = x[85:125]
# y4 = y[85:125]
# x5 = x[115:150]
# y5 = y[115:150]

x1 = x[0:35]
x1 = np.concatenate((x[-5: 1], x1), axis=0)
y1 = y[0:35]
y1 = np.concatenate((y[-5: 1], y1), axis=0)
x2 = x[30:65]
y2 = y[30:65]
x3 = x[60:95]
y3 = y[60:95]
x4 = x[90:125]
y4 = y[90:125]
x5 = x[120:155]
y5 = y[120:155]

popt1, pcov1 = curve_fit(polynomic_func_9, x1, y1, maxfev=10000)
popt2, pcov2 = curve_fit(polynomic_func_9, x2, y2, maxfev=10000)
popt3, pcov3 = curve_fit(polynomic_func_9, x3, y3, maxfev=10000)
popt4, pcov4 = curve_fit(polynomic_func_9, x4, y4, maxfev=10000)
popt5, pcov5 = curve_fit(polynomic_func_9, x5, y5, maxfev=10000)
inter = time.time()
print("computation time0: ", inter - start)
# popt1, pcov1 = curve_fit(multi_periodic_func_5_, x, y, maxfev=1000000000)
# end = time.time()
# print("computation time: ", end - start)

# test_points = np.array([x[:10], y[:10]]).T
# bezier_fit(test_points)

# bezier fitting

x1 = x[0:35]

# retrive the starshaped polygon
y1 = polynomic_func_9(x1, *popt1)
y2 = polynomic_func_9(x2, *popt2)
y3 = polynomic_func_9(x3, *popt3)
y4 = polynomic_func_9(x4, *popt4)
y5 = polynomic_func_9(x5, *popt5)

# plot the starshaped points and the fitted periodic function
plt.plot(x, y, 'ko', label="Original Noised Data")
plt.plot(x1, y1, 'r-', label="Fitted Curve poly")
plt.plot(x2, y2, 'g-', label="Fitted Curve poly")
plt.plot(x3, y3, 'b-', label="Fitted Curve poly")
plt.plot(x4, y4, 'y-', label="Fitted Curve poly")
plt.plot(x5, y5, 'b-', label="Fitted Curve poly")
# plt.plot(x, multi_periodic_func_5_(x, *popt1), 'b-', label="Fitted Curve 5")
plt.legend()
plt.show()

# retrive the starshaped polygon
re_starshaped_points = []
theta_list = np.linspace(0, 2 * np.pi, 150)

for i in range(len(theta_list)):
    if i < 31:
        points = [y1[i] * np.cos(theta_list[i]), y1[i] * np.sin(theta_list[i])]
    elif i > 30 and i < 61:
        points = [y2[i-31] * np.cos(theta_list[i]), y2[i-31] * np.sin(theta_list[i])]
    elif i > 60 and i < 91:
        points = [y3[i-61] * np.cos(theta_list[i]), y3[i-61] * np.sin(theta_list[i])]
    elif i > 90 and i < 121:
        points = [y4[i-91] * np.cos(theta_list[i]), y4[i-91] * np.sin(theta_list[i])]
    else:
        points = [y5[i-121] * np.cos(theta_list[i]), y5[i-121] * np.sin(theta_list[i])]
    re_starshaped_points.append(points)

# plot the re_starshaped_points
re_starshaped_points = np.array(re_starshaped_points)
plt.plot(re_starshaped_points[0:-5, 0], re_starshaped_points[0:-5, 1], 'r--', label='fit')
# plot the starshaped points
plt.plot(starshaped_points[0:-1, 0], starshaped_points[0:-1, 1], 'b--', label='raw')
# plot center
plt.plot(0, 0, 'ko', label='center')
plt.axis('equal')
plt.legend()
plt.show()

