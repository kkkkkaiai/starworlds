import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from obstacles import StarshapedPolygon
from utils import generate_star_polygon

def polynomic_func_9(x, a, b, c, d, e, f, g, h, i, j):
    return a * x**9 + b * x**8 + c * x**7 + d * x**6 + e * x**5 + f * x**4 + g * x**3 + h * x**2 + i * x + j

def polynomic_func_11(x, a, b, c, d, e, f, g, h, i, j, k, l):
    return a * x**11 + b * x**10 + c * x**9 + d * x**8 + e * x**7 + f * x**6 + g * x**5 + h * x**4 + i * x**3 + j * x**2 + k * x + l

class PolygonFit:
    def __init__(self, points, center, poly_degree=9, poly_num=11, padding=5):
        self._center = center
        self._points = np.array(points)
        self._points_num = self._points.shape[0]
        self._poly_degree = poly_degree
        self._poly_num = poly_num
        
        self._polar_angle = np.array([np.arctan2(points[i][1], points[i][0]) for i in range(len(points))])
        self._polar = np.array([np.linalg.norm(points[i] - center) for i in range(len(points))])
        self._polar = gaussian_filter1d(self._polar, sigma=3)
        self._padding = padding

        self._robot_radius = 0.5
        self._max_distance = 4

        self._points_divide_index = {'points_index':[], 'angle_index':[]}
        self._fit_parameter = []
        self.divide_points()

    def divide_points(self):
        segment_length = int(self._points_num / self._poly_num)
        angle_diff = 2 * np.pi / self._poly_num
        for i in range(self._poly_num):
            self._points_divide_index['points_index'].append(i * segment_length)
            angle = i * angle_diff
            # if abs(self._points[i * segment_length][0] - 0.0) < 1e-6:
            #     if i > self._poly_num / 2:
            #         angle = np.pi
            # else:
            #     angle = np.arctan2(self._points[i * segment_length][1], self._points[i * segment_length][0])
            # if angle < 0:
            #     angle += 2 * np.pi
            # if i != 0 and abs(angle-0.0) < 1e-6:
            #     angle = 2 * np.pi
            self._points_divide_index['angle_index'].append(angle)

    def radian_to_index(self, angle):
        # angle should be radian   
        if angle < 0:
            angle += 2 * np.pi 
        for i in range(self._poly_num):
            if angle < self._points_divide_index['angle_index'][i]:
                return i - 1
        return self._poly_num - 1


    def predict_gamma_angle(self, angle):
        index = self.radian_to_index(angle)
        if self._poly_degree == 9:
            gamma = polynomic_func_9(angle, *self._fit_parameter[index])
        elif self._poly_degree == 11:
            gamma = polynomic_func_11(angle, *self._fit_parameter[index])
        # print(angle, index, gamma)
        return gamma

    def fit(self):
        padding = self._padding
        for i in range(self._poly_num):
            if i == self._poly_num - 1:
                # add padding points to get a better fit parameter
                x = self._polar_angle[self._points_divide_index['points_index'][i]:self._points_num+padding]
                y = self._polar[self._points_divide_index['points_index'][i]:self._points_num+padding]
            else:
                x = self._polar_angle[self._points_divide_index['points_index'][i]:self._points_divide_index['points_index'][i + 1]+padding]
                y = self._polar[self._points_divide_index['points_index'][i]:self._points_divide_index['points_index'][i + 1]+padding]
            
            if self._poly_degree == 9:
                # check if the points are enough to fit
                if len(x) < 9:
                    x = np.concatenate((x[i-5: i-1], x), axis=0)
                    y = np.concatenate((y[i-5: i-1], y), axis=0)
                popt, pcov = curve_fit(polynomic_func_9, x, y)
            elif self._poly_degree == 11:
                if len(x) < 11:
                    x = np.concatenate((x[i-7: i-1], x), axis=0)
                    y = np.concatenate((y[i-7: i-1], y), axis=0)
                popt, pcov = curve_fit(polynomic_func_11, x, y)
            self._fit_parameter.append(popt)
        print(self._points_divide_index)
        # print(self._fit_parameter)

    def plot_polar(self):
        x = self._polar_angle
        y = self._polar.flatten()
        plt.scatter(x, y, c='b', s=20, label='original')

        x_pred = []
        y_pred = []
        for i in range(self._poly_num):
            if i == self._poly_num - 1:
                x = self._polar_angle[self._points_divide_index['points_index'][i]:]
            else:
                x = self._polar_angle[self._points_divide_index['points_index'][i]:self._points_divide_index['points_index'][i + 1]]
            
            x_pred.append(x)
            if self._poly_degree == 9:
                y_pred.append(polynomic_func_9(x, *self._fit_parameter[i]))
            elif self._poly_degree == 11:
                y_pred.append(polynomic_func_11(x, *self._fit_parameter[i]))
        
        
        merge_x_pred = np.concatenate(x_pred)
        merge_y_pred = np.concatenate(y_pred)
        plt.scatter(merge_x_pred, merge_y_pred, c='r', s=20, label='fit')
        
        plt.title('Radius of starshaped polygon')
        plt.legend()
        plt.show()

    def plot_retrive_points(self):
        plt.plot(self._points[:, 0], self._points[:, 1], 'b--', label='original')
        retrieve_points = []
        color_list = []
        
        x = np.linspace(-np.pi, np.pi, self._points_num)
        for i in range(x.shape[0]):
            distance = self.predict_gamma_angle(x[i])
            index = self.radian_to_index(x[i])
            if distance > self._max_distance or distance < 0:
                distance = self._robot_radius
 
            retrieve_points.append([distance * np.cos(x[i]), distance * np.sin(x[i])])
            color_list.append(index)
        retrieve_points = np.array(retrieve_points)

        # convert number to color
        color_list = np.array(color_list)
        color_list = color_list / self._poly_num
        plt.scatter(retrieve_points[:, 0], retrieve_points[:, 1], c=color_list, cmap='jet', s=20, label='fit')
        plt.scatter(0, 0, c='k', s=20, label='center')
        plt.axis('equal')
        plt.title('Raw points and fit points')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    # generate points
    avg_radius = 2
    starshaped_center = [0, 0]
    starshaped_edge = generate_star_polygon(starshaped_center, avg_radius, irregularity=0.1, spikiness=0.5, num_vertices=10)
    pol = StarshapedPolygon(starshaped_edge, xr=np.array(starshaped_center))
    starshaped_points = []
    for i in np.linspace(0, 2 * np.pi, 500):
        x = pol.xr() + 100*np.array([np.cos(i), np.sin(i)])
        b = pol.boundary_mapping(x)
        starshaped_points.append(b)

    pf = PolygonFit(starshaped_points, starshaped_center)
    pf.fit()
    pf.plot_polar()
    pf.plot_retrive_points()
