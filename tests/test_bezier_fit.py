import matplotlib.pyplot as plt
import numpy as np
import math

def input(name):
    f = open(name, 'r')
    data = f.readlines()
    f.close()
    xy = []
    for row in data:
        row = row.replace('\n', '')
        # print(row)
        tmp_list = row.split('\t')
        # print(tmp_list)
        xy.append([float(tmp_list[0]), float(tmp_list[1])])
    points = np.array(xy)
    return points

def plot_composite_bezier_curve(P):
    n = 2
    nt = 200
    ts = np.linspace(0., 1., nt)

    Q = np.zeros((nt, 2))
    for i,t in enumerate(ts):
        # Q[i,:] = point_on_bezier_curve(P,t)
        Q[i, :] = BezierCurvePoint(P, t)

    plt.plot(Q[:,0], Q[:,1], '-r',label='Bezier Fitting Curve')
    plt.plot(P[:,0], P[:,1], '--', linewidth=0.2)

def iBersteinItem(i,n,t):
    B_i_n_t=math.factorial(n)/math.factorial(i)/math.factorial(n-i)*t**(i)*(1-t)**(n-i)
    return B_i_n_t

def BezierCurvePoint(P,t):
    n = len(P) - 1
    c = 0.
    for k in range(0, n + 1):
        iB=iBersteinItem(k,n,t)
        c += iB * P[k]
    return c

def bezier_fit(points):
    x=points[:,0]
    y=points[:,1]
    pn = len(x)
    pdistance=np.zeros(pn-1)
    for i in range(pn):
        if i==0:
            pdistance[i] = math.sqrt((x[i] - x[i + 1]) ** 2 + (y[i] - y[i + 1]) ** 2)
        if i<pn-1 and i>0:
            pdistance[i]=math.sqrt((x[i]-x[i+1])**2+(y[i]-y[i+1])**2)+pdistance[i-1]
    pdistance=pdistance/pdistance.max()
    t_input=np.zeros(1)
    t_input=np.append(t_input,pdistance)

    # ts = np.linspace(0., 1., pn+1)
    ts=t_input
    # print(ts)
    B=np.zeros([pn,pn])
    for i in range(pn):
        for index,t in enumerate(ts):
            # print(index, i,t, B[index, i])
            B[index,i]=iBersteinItem(i,pn-1,t)
            # print(index, i,t, B[index, i])
    # print(B)
    Pcontrol=np.linalg.lstsq(B, points, rcond=-1)
    Pcontrol=Pcontrol[0]
    print('Bezier Control Points:\n')
    for item in Pcontrol:
        print('{:.4f},{:.4f}'.format(item[0],item[1]))
    plt.plot(Pcontrol[:,0],Pcontrol[:,1],'o',label='Bezier Control Points')


    #plot the two curves
    plt.plot(x, y, 'bo', label='Input')

    plot_composite_bezier_curve(Pcontrol)

    plt.legend()
    plt.show()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    points=input('input.dat')
    # print(points)
    bezier_fit(points)
   