import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib
from scipy import integrate

matplotlib.rcParams['font.sans-serif'] = ['STSong']
matplotlib.rcParams['axes.unicode_minus'] = False

class LegendreSeriesApproximation:
    """
    勒让德级数逼近函数：符号运算与数值运算
    """
    def __init__(self, fun, x_span=np.array([-1,1]), k=6):
        """
        必要的参数初始化
        """
        self.a, self.b = x_span[0], x_span[1]   # 区间左右端点
        self.fun_transform, self.lambda_fun = self.internal_transform(fun)      # 区间转化函数
        self.k = k      # 逼近已知函数所需项数
        self.T_coefficient = None       # 勒让德各项和对应系数
        self.approximation_poly = None      # 逼近多项式
        self.poly_coefficient = None        # 逼近多项式系数
        self.polynomial_orders = None       # 逼近多项式各项阶数
        self.max_abs_error = np.infty       # 逼近多项式的最大绝对误差
        self.mae = np.infty     # 随机数模拟的绝对误差均值
    
    def internal_transform(self, fun):
        """
        区间转化函数
        """
        t = fun.free_symbols.pop()  # 获取函数的符号变量
        fun_transform = fun.subs(t, (self.b - self.a) / 2 * t + (self.b + self.a) / 2)  # 区间变换
        lambda_fun = sp.lambdify(t, fun)  # 构成lambda函数
        return fun_transform, lambda_fun
    
    def fit_approximation(self):
        """
        勒让德级数逼近核心算法:递推Pn(x),求解系数fn，构成逼近多项式
        """
        t = self.fun_transform.free_symbols.pop()  # 获取函数的符号变量
        term = sp.Matrix.zeros(self.k+1,1)
        term[0], term[1] = 1, t  # 初始化第一项和第二项
        coefficient = np.zeros(self.k+1)    # 勒让德级数多项式系数
        expr = sp.lambdify(t, term[0] * self.fun_transform)
        coefficient[0] = integrate.quad(expr, -1, 1)[0]/2  # f0系数
        expr = sp.lambdify(t, term[1] * self.fun_transform)
        coefficient[1] = integrate.quad(expr, -1, 1)[0]*3/2  # f1系数
        self.approximation_poly = coefficient[0] + coefficient[1]*term[1]
        # 从第三项开始循环求解
        for i in range(2, self.k+1):
            term[i] = sp.expand(((2*i-1)*t*term[i-1]-(i-1)*term[i-2])/i)     # 递推项Pn(x)
            expr = sp.lambdify(t, term[i]*self.fun_transform)
            coefficient[i] = integrate.quad(expr, -1, 1, full_output=1, points=[-1,1])[0]*((2*i+1)/2)  # fn系数
            self.approximation_poly += coefficient[i]*term[i]
        
        self.T_coefficient = [term, coefficient]    # 存储勒让德各项和对应系数
        self.approximation_poly = sp.expand(self.approximation_poly)
        polynomial = sp.Poly(self.approximation_poly, t)
        self.poly_coefficient = polynomial.coeffs()
        self.polynomial_orders = polynomial.monoms()
        self.error_analysis()   # 误差分析

    def cal_x0(self, x0):
        """
        求解给定点的逼近值
        x0：所求逼近点的x坐标
        """
        from utils import orthogonal_polynomial_utils
        return orthogonal_polynomial_utils.cal_x0(self.approximation_poly, x0, self.a, self.b)
    
    def error_analysis(self):
        """
        勒让德级数逼近度量
        进行10次模拟，每次模拟指定区间随机生成100个数据点，然后根据度量方法分析
        """
        from utils import orthogonal_polynomial_utils
        params = self.approximation_poly, self.lambda_fun, self.a, self.b
        self.max_abs_error, self.mae = orthogonal_polynomial_utils.error_analysis(params)
    
    def plt_approximation(self, is_show=True):
        """
        绘制逼近多项式的图像
        """
        from utils import orthogonal_polynomial_utils
        params = self.approximation_poly, self.lambda_fun, self.a, self.b, self.k, self.mae, 'Legendre Series', is_show
        orthogonal_polynomial_utils.plt_approximation(params)

# 勒让德级数逼近
if __name__ == "__main__":
    t = sp.symbols('t')
    fun = sp.exp(t)

    lsa = LegendreSeriesApproximation(fun,x_span=[-1,1],k=3)
    lsa.fit_approximation()
    print('勒让德级数逼近系数及对应阶次：')
    print(lsa.poly_coefficient)
    print(lsa.polynomial_orders)
    print('勒让德级数逼近多项式：')
    print(lsa.approximation_poly)
    print('勒让德级数逼近的最大绝对误差是：')
    print(lsa.max_abs_error)
    print(lsa.T_coefficient)
    lsa.plt_approximation()