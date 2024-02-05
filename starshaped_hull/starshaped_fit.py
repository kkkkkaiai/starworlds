import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from sklearn.cluster import DBSCAN

def polynomic_func_9(x, a, b, c, d, e, f, g, h, i, j):
    return a * x**9 + b * x**8 + c * x**7 + d * x**6 + e * x**5 + f * x**4 + g * x**3 + h * x**2 + i * x + j

class StarshapedRep:
    def __init__(self, points, center, theta=0.0) -> None:
        self.popt1 = self.popt2 = self.popt3 = self.popt4 = self.popt5 = None
        self.pcov1 = self.pcov2 = self.pcov3 = self.pcov4 = self.pcov5 = None

        self.points = points.T
        self.center = self.reference_point = center

        self.theta = theta
        self.starshaped_fit()
        self.get_frontier_points()

        # self.draw()

    def starshaped_fit(self, segment=5, padding=5):
        starshaped_range = np.array([np.linalg.norm(self.points[i] - self.center) for i in range(len(self.points))])

        # gaussian smooth
        self.starshaped_range = gaussian_filter1d(starshaped_range, sigma=3)

        self.x = np.linspace(0, 2 * np.pi, len(self.starshaped_range))
        self.y = self.starshaped_range.flatten()  

        # segment data to 5 parts
        segment_num = len(starshaped_range) // segment
        self.x1 = self.x[0:segment_num]
        self.x1 = np.concatenate((self.x[-padding: 1], self.x1), axis=0)
        self.y1 = self.y[0:segment_num]
        self.y1 = np.concatenate((self.y[-padding: 1], self.y1), axis=0)
        self.x2 = self.x[segment_num:segment_num*2+padding]
        self.y2 = self.y[segment_num:segment_num*2+padding]
        self.x3 = self.x[segment_num*2:segment_num*3+padding]
        self.y3 = self.y[segment_num*2:segment_num*3+padding]
        self.x4 = self.x[segment_num*3:segment_num*4+padding]
        self.y4 = self.y[segment_num*3:segment_num*4+padding]
        self.x5 = self.x[segment_num*4:segment_num*5+padding]
        self.y5 = self.y[segment_num*4:segment_num*5+padding]

        self.popt1, self.pcov1 = curve_fit(polynomic_func_9, self.x1, self.y1, maxfev=10000)
        self.popt2, self.pcov2 = curve_fit(polynomic_func_9, self.x2, self.y2, maxfev=10000)
        self.popt3, self.pcov3 = curve_fit(polynomic_func_9, self.x3, self.y3, maxfev=10000)
        self.popt4, self.pcov4 = curve_fit(polynomic_func_9, self.x4, self.y4, maxfev=10000)
        self.popt5, self.pcov5 = curve_fit(polynomic_func_9, self.x5, self.y5, maxfev=10000)

    def get_frontier_points(self, 
                            robot_radius=0.8, 
                            min_samples=2, # dbscan parameter
                            ):
        dbscan = DBSCAN(eps=robot_radius, min_samples=min_samples)
        filter_points = np.array(self.points)
        labels = dbscan.fit_predict(filter_points)
        self.labels = labels

        label_list = []
        for i in range(np.max(labels)+1):
            label_list.append([])
        for i in range(len(labels)):
            if labels[i] != -1:
                label_list[labels[i]].append(filter_points[i])

        # in the label_list, the points is sorted by the angle 
        # find the points on the both sides, 
        sides_points = []
        sorted_points = []
        for i in range(len(label_list)):
            # calculate the angle of all points
            points = np.array(label_list[i])
            angles = np.arctan2(points[:, 1] - self.center[1], points[:, 0] - self.center[0])

            # print(sorted_angle)
            min_angle = np.min(angles)
            max_angle = np.max(angles)
            print('angle', min_angle, max_angle)

            if max_angle - min_angle > np.pi*1.9:
                # find the max angle diff
                print('in')
                max_diff = 0
                temp_max_idx = 0

                print(angles)

                for j in range(len(angles)-1):
                    diff = abs(angles[j+1] - angles[j])
                    print(diff, angles[j+1], angles[j])

                    # if the diff is larger than pi, it means the angle is from 2pi to 0
                    if diff > np.pi:
                        diff = 2*np.pi - diff

                    if diff > max_diff:
                        max_diff = diff
                        temp_max_idx = j
                
                # check the last on and the first one
                diff = abs(angles[0] - angles[-1])
                if diff > np.pi:
                    diff = 2*np.pi - diff

                if diff > max_diff:
                    max_diff = diff
                    temp_max_idx = len(angles) - 1

                min_idx = temp_max_idx + 1
                if min_idx >= len(angles):
                    min_idx = 0
                max_idx = temp_max_idx
            else:
                # find the max and min angle
                max_idx = np.argmax(angles)
                min_idx = np.argmin(angles)
            
            print('idx', max_idx, min_idx)

            # the original index in the filter_points
            # max_idx = np.where(np.all(filter_points == points[max_idx], axis=1))[0][0]
            # min_idx = np.where(np.all(filter_points == points[min_idx], axis=1))[0][0]

            # put the points on the both sides into the sides_points
            sides_points.append([max_idx, min_idx])
            sorted_points.append(label_list[i][min_idx])
            sorted_points.append(label_list[i][max_idx])

        # the frontier point is the middle point on the line of the adjecent sides points
        sorted_points = np.array(sorted_points)
        frontier_points = []

        for i in range(1, len(sorted_points), 2):
            left = i
            right = i+1
            if right >= len(sorted_points):
                right = 0
            frontier_points.append((sorted_points[left] + sorted_points[right]) / 2)

        self.frontier_points = np.array(frontier_points)
        self.sorted_points = np.array(sorted_points)
        # print(sorted_points)
        # print(frontier_points)


    def radius(self, theta):
        if theta < self.x1[-1]:
            return polynomic_func_9(theta, *self.popt1)
        elif theta < self.x2[-1]:
            return polynomic_func_9(theta, *self.popt2)
        elif theta < self.x3[-1]:
            return polynomic_func_9(theta, *self.popt3)
        elif theta < self.x4[-1]:
            return polynomic_func_9(theta, *self.popt4)
        else:
            return polynomic_func_9(theta, *self.popt5)
    
    def get_gamma(self, point, inverted=True, p=1):
        theta = np.arctan2(point[1] - self.center[1], point[0] - self.center[0])
        if theta < 0:
            theta += 2 * np.pi
        radius_val = self.radius(theta)
        dis_from_center = np.linalg.norm(point - self.center)
        # debug
        # print('gamma', radius_val, dis_from_center)
        if inverted:
            return (radius_val / dis_from_center)**(2*p)
        else:
            return (dis_from_center / radius_val)**(2*p)
        
    def is_in_star(self, point, inverted=True):
        gamma = self.get_gamma(point, inverted=inverted)
        # debug
        # print(gamma)
        return gamma > 1

    def draw(self, name='test'):

        y1 = polynomic_func_9(self.x1, *self.popt1)
        y2 = polynomic_func_9(self.x2, *self.popt2)
        y3 = polynomic_func_9(self.x3, *self.popt3)
        y4 = polynomic_func_9(self.x4, *self.popt4)
        y5 = polynomic_func_9(self.x5, *self.popt5)

        # plt.plot(self.x1, y1, 'r-', label="Fitted Curve poly")
        # plt.plot(self.x2, y2, 'g-', label="Fitted Curve poly")
        # plt.plot(self.x3, y3, 'b-', label="Fitted Curve poly")
        # plt.plot(self.x4, y4, 'y-', label="Fitted Curve poly")
        # plt.plot(self.x5, y5, 'b-', label="Fitted Curve poly")

        # print(self.x1, self.x2, self.x3, self.x4, self.x5)

        # plt.plot(self.x, self.y, 'ko', label="Fitted Curve poly")
        # plt.savefig("starshaped_fit.png", dpi=300)

        # plt.cla()

        re_starshaped_points = []
        theta_list = np.linspace(0, 2 * np.pi, len(self.starshaped_range), endpoint=False)
        for theta in theta_list:
            # convert to cartesian coordinates and add bias
            re_starshaped_points.append([self.radius(theta) * np.cos(theta), self.radius(theta) * np.sin(theta)]+ self.center) 
        re_starshaped_points = np.array(re_starshaped_points)

        # generate a random color
        color = np.random.rand(3,)
        # plt.plot(re_starshaped_points[0:, 0], re_starshaped_points[0:, 1], color, label='fit')
        # plot the starshaped points
        # plt.scatter(self.points[0:-1, 0], self.points[0:-1, 1], label='raw')
        # plot dbscan result
        plt.scatter(self.points[:, 0], self.points[:, 1], c=self.labels, label='dbscan', s=5)
        # plt.scatter(self.points[:, 0], self.points[:, 1], c=color, label='dbscan', s=5)
    
        # plot the center point
        plt.scatter(self.center[0], self.center[1], label='center', s=10)

        # plot frontier points as plus symbol
        plt.plot(self.frontier_points[:, 0], self.frontier_points[:, 1], 'k+', label='frontier points')
        # scatter frontier points with color and plus symbol
        plt.scatter(self.frontier_points[:, 0], self.frontier_points[:, 1], c=color, label='frontier points', s=30)
        # the sides points
        plt.plot(self.sorted_points[:, 0], self.sorted_points[:, 1], 'y+', label='sorted points', scalex=20, scaley=20)
        plt.axis('equal')
        # plt.savefig(name+'_starshaped_fit_retrive.png', dpi=300)







