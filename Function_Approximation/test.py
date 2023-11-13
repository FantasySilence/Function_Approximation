import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib
from ChebyshevSeries import ChebyshevSeriesApproximation
from ChebyshevZero import ChebyshevZeroPointsInterpolation
from LegendreSeries import LegendreSeriesApproximation
from LeastSquareFitting import LeastSquarePolynomialFitting

matplotlib.rcParams['font.sans-serif'] = ['STSong']
matplotlib.rcParams['axes.unicode_minus'] = False

# 误差分析，以切比雪夫级数逼近为例
if __name__ == "__main__":
    x = sp.symbols('x')
    fun = sp.exp(x)

    orders = np.arange(4,21,1)
    mae_ = np.zeros(len(orders))
    for i, order in enumerate(orders):
        print(order)
        czpi = ChebyshevSeriesApproximation(fun,x_span=[0,1],k=order)
        czpi.fit_approximation()
        mae_[i] = czpi.mae
    
    plt.figure(figsize=(8,6))
    plt.plot(orders,mae_,'ro-',lw=1.5)
    plt.xlabel('Orders',fontdict={"fontsize":12})
    plt.ylabel('Mean Abs Error',fontdict={"fontsize":12})
    plt.title("Absolute Error Variation Curve with Different Orders",fontdict={"fontsize":14})
    plt.grid(ls=":")
    plt.show()

# 三种逼近方法的比较
if __name__ == "__main__":
    def runge_fun(x):
        return 1/(x**2+1)

    t = sp.symbols('t')
    fun = 1/(t**2+1)
    orders = [10, 25]
    plt.figure(figsize=(14,15))
    for i, order in enumerate(orders):
        plt.subplot(321 + i)
        cpzi = ChebyshevZeroPointsInterpolation(runge_fun,x_span=[-5,5],order=order)
        cpzi.fit_approximation()
        cpzi.plt_approximation(is_show=False)
        print("切比雪夫零点插值的最大绝对误差：",cpzi.max_abs_error)
        plt.subplot(323 + i)
        csa = ChebyshevSeriesApproximation(fun, x_span=[-5,5], k=order)
        csa.fit_approximation()
        csa.plt_approximation(is_show=False)
        print("切比雪夫级数逼近的最大绝对误差：",csa.max_abs_error)
        plt.subplot(325 + i)
        lsa = LegendreSeriesApproximation(fun, x_span=[-5,5], k=order)
        lsa.fit_approximation()
        lsa.plt_approximation(is_show=False)
        print("勒让德级数逼近的最大绝对误差：",lsa.max_abs_error)
    plt.show()

# 最小二乘曲线拟合不同阶次对比
if __name__ == "__main__":
    x = np.linspace(0,5,100)
    np.random.seed(0)
    y = np.sin(x)*np.exp(-x) + np.random.randn(100)/100
    xi = np.linspace(0,5,100)
    plt.figure(figsize=(8,6))
    orders = [12,15,17,20,25]
    line_style = ["--",":","-","-.","-*"]
    for k,line in zip(orders,line_style):
        ls = LeastSquarePolynomialFitting(x,y,k=k)
        ls.fit_curve()
        yi = ls.cal_x0(xi)
        plt.plot(xi,yi,line,lw=1.5,label="Order=%d, MSE=%.2e"%(k,ls.mse))
    plt.plot(x,y,'ko',label="离散数据点")
    plt.xlabel('X',fontdict={"fontsize":12})
    plt.ylabel("Y",fontdict={"fontsize":12})
    plt.title("Least Square Fitting with Different Orders",fontdict={"fontsize":14})
    plt.legend(loc='best')
    plt.show()
