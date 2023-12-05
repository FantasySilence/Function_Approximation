# <font face="黑体">函数逼近各方法的调用说明</font>

<font face="Times New Roman">Powered by: </font><font face="宋体">御河DE天街</font>

## <font face="Times New Roman">1 </font><font face="黑体">最小二乘逼近</font>

### <font face="Times New Roman">1.1 </font><font face="黑体">初始化对象</font>

<font face="宋体">首先需要导入对应的模块：</font><br/>

```
from Function_Approximation.LeastSquareFitting import LeastSquarePolynomialFitting
```

#### <font face="Times New Roman">1.1.1 </font><font face="黑体">参数说明</font><br/>
```
LeastSquarePolynomialFitting(x, y, k=None, w=None, base_fun='default', fun_list=None)
```

<font face="宋体">其中，<code>x,y</code>为离散数据点，需要注意的是，离散数据点的长度必须一致，建议以<code>numpy.array</code>的形式传入。<code>w</code>为权重，权重默认为<font face="Times New Roman">1</font>，如果需要另外传入权重，注意权重数组(建议以<code>numpy.array</code>的形式传入)的长度同样需要与离散数据点的长度保持一致。<code>base_fun</code>为基函数类型，默认为<code>default</code>，使用<font face="Times New Roman">1,  $\displaystyle x$, $\displaystyle x^{2}$ ... </font>等幂函数作为基函数。另外还有<code>ort</code>，使用正交多项式作为基函数；<code>other</code>，使用自定义的一系列函数作为基函数。当<code>base_fun</code>为<code>default</code>或<code>ort</code>时，需要指定多项式的最高阶次<code>k</code>；当<code>base_fun</code>为<code>other</code>是，需要给出自定义的基函数列表(采用符号定义，且为单项式函数)。</font><br/>

#### <font face="Times New Roman">1.1.2 </font><font face="黑体">方法(类属性)简介</font><br/>

<font face="宋体">进行拟合：<code>self.fit_curve()</code><br/>绘制拟合后的曲线：<code>self.plt_curve_fit()</code><br/>拟合的结果：<code>self.fit_poly</code><br/>
如果多项式拟合，拟合多项式的系数以及对应的阶次：<code>self.poly_coefficient</code>和<code>self.polynomial_orders</code><br/>

### <font face="Times New Roman">1.2 </font><font face="黑体">调用示例</font>
#### <font face="Times New Roman">2.1.1 </font><font face="黑体">使用幂函数作为基函数进行拟合</font><br/>

```
import numpy as np
from Function_Approximation.LeastSquareFitting import LeastSquarePolynomialFitting
x = np.linspace(0,5,15)
np.random.seed(0)
y = 2*np.sin(x)*np.exp(-x)+np.random.randn(15)/100
ls = LeastSquarePolynomialFitting(x,y,k=5)
ls.fit_curve()
print('拟合多项式为：{}'.format(ls.fit_poly))
print('拟合多项式系数：{}'.format(ls.poly_coefficient))
print('拟合多项式系数的阶次：{}'.format(ls.polynomial_orders))
ls.plt_curve_fit()
``` 

#### <font face="Times New Roman">2.1.2 </font><font face="黑体">使用正交多项式作为基函数进行拟合</font>

```
import numpy as np
from Function_Approximation.LeastSquareFitting import LeastSquarePolynomialFitting
x = np.array([0, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
y = np.array([1, 1.75, 1.96, 2.19, 2.44, 2.71, 3])
ls2 = LeastSquarePolynomialFitting(x,y,k=2,base_fun='ort')
ls2.fit_curve()
print('拟合多项式：{}'.format(ls2.fit_poly))
print('拟合多项式系数：{}'.format(ls2.poly_coefficient))
print('拟合多项式系数的阶次：{}'.format(ls2.polynomial_orders))
ls2.plt_curve_fit()
```

#### <font face="Times New Roman">2.1.3 </font><font face="黑体">使用自定义的一系列函数作为基函数进行拟合</font>

```
import numpy as np
import sympy as sp
from Function_Approximation.LeastSquareFitting import LeastSquarePolynomialFitting
t = sp.Symbol('t')
fun_list = [1, sp.log(t), sp.cos(t), sp.exp(t)]
x = np.array([0.24, 0.65, 0.95, 1.24, 1.73, 2.01, 2.23, 2.52, 2.77, 2.99])
y = np.array([0.23, -0.26, -1.1, -0.45, 0.27, 0.1, -0.29, 0.24, 0.56, 1])
ls1 = LeastSquarePolynomialFitting(x,y,base_fun='other',fun_list=fun_list)
ls1.fit_curve()
print('拟合多项式为：{}'.format(ls1.fit_poly))
print('拟合多项式系数：{}'.format(ls1.poly_coefficient))
ls1.plt_curve_fit()
```

## <font face="Times New Roman">2 </font><font face="黑体">自适应三次样条逼近</font>

### <font face="Times New Roman">2.1 </font><font face="黑体">初始化对象</font> 

<font face="宋体">首先需要导入对应的模块：</font><br/>

```
from Function_Approximation.AdaptiveSpline import AdaptiveSplineApproximation
```

#### <font face="Times New Roman">2.1.1 </font><font face="黑体">参数说明</font><br/>
```
AdaptiveSplineApproximation(fun, x_span, eps=1e-5)
```

<font face="宋体">其中，```fun```为被逼近的函数，```x_span```为逼近的区间，```eps```为精度，默认为1e-5。

#### <font face="Times New Roman">2.1.2 </font><font face="黑体">方法(类属性)简介</font><br/>

进行拟合：```self.fit_approximation()```<br/>
绘制拟合曲线：```self.plt_approximation(is_show=True)```<br/>

### <font face="Times New Roman">2.2 </font><font face="黑体">调用示例</font>

```
from Function_Approximation.AdaptiveSpline import AdaptiveSplineApproximation

def fun(x):
    return 1/(1+x**2)

ase = AdaptiveSplineApproximation(fun,x_span=[-5,5],eps=1e-8)
ase.fit_approximation()
print(ase.node)
print(ase.spline_obj.m)
ase.plt_approximation(is_show=True)
```

## <font face="Times New Roman">3 </font><font face="黑体">最佳平方逼近</font>

### <font face="Times New Roman">3.1 </font><font face="黑体">初始化对象</font> 

<font face="宋体">首先需要导入对应的模块：</font><br/>

```
from Function_Approximation.BestSquare import BestSquareApproximation
```

#### <font face="Times New Roman">3.1.1 </font><font face="黑体">参数说明</font><br/>
```
BestSquareApproximation(fun, x_span, k)
```

<font face="宋体">其中，```fun```为被逼近的函数，```x_span```为逼近的区间，```k```为拟合多项式的最高阶次。

#### <font face="Times New Roman">3.1.2 </font><font face="黑体">方法(类属性)简介</font><br/>

进行拟合：```self.fit_approximation()```<br/>
绘制拟合曲线：```self.plt_approximation(is_show=True)```<br/>
逼近多项式系数及对应阶次：```self.poly_coefficient```和```self.polynomial_orders```<br/>
逼近多项式: ```self.approximation_poly```<br/>
逼近的最大绝对误差: ```self.max_abs_error```<br/>

### <font face="Times New Roman">3.2 </font><font face="黑体">调用示例</font>

```
import sympy as sp
from Function_Approximation.BestSquare import BestSquareApproximation

t = sp.symbols('t')
fun = 1/(1+t**2)

bsa = BestSquareApproximation(fun,x_span=[-5,5],k=30)
bsa.fit_approximation()
print('最佳平方逼近系数及对应阶次：')
print(bsa.poly_coefficient)
print(bsa.polynomial_orders)
print('最佳平方逼近多项式：')
print(bsa.approximation_poly)
print('最佳平方逼近的最大绝对误差是：')
print(bsa.max_abs_error)
bsa.plt_approximation()
```

## <font face="Times New Roman">4 </font><font face="黑体">切比雪夫级数逼近</font>

### <font face="Times New Roman">4.1 </font><font face="黑体">初始化对象</font> 

<font face="宋体">首先需要导入对应的模块：</font><br/>

```
from Function_Approximation.ChebyshevSeries import ChebyshevSeriesApproximation
```

#### <font face="Times New Roman">4.1.1 </font><font face="黑体">参数说明</font><br/>

```
BestSquareApproximation(fun, x_span=np.array([-1,1]), k=6)
```

<font face="宋体">其中，```fun```为被逼近的函数，```x_span```为逼近的区间，默认为[-1,1]```k```为拟合多项式的最高阶次，默认为6。

#### <font face="Times New Roman">4.1.2 </font><font face="黑体">方法(类属性)简介</font><br/>

进行拟合：```self.fit_approximation()```<br/>
绘制拟合曲线：```self.plt_approximation(is_show=True)```<br/>
逼近多项式系数及对应阶次：```self.poly_coefficient```和```self.polynomial_orders```<br/>
逼近多项式: ```self.approximation_poly```<br/>
逼近的最大绝对误差: ```self.max_abs_error```<br/>

### <font face="Times New Roman">4.2 </font><font face="黑体">调用示例</font>

```
import sympy as sp
from Function_Approximation.ChebyshevSeries import ChebyshevSeriesApproximation

t = sp.symbols('t')
fun = sp.exp(t)

csa = ChebyshevSeriesApproximation(fun,x_span=[-1,1],k=3)
csa.fit_approximation()
print('切比雪夫级数逼近系数及对应阶次：')
print(csa.poly_coefficient)
print(csa.polynomial_orders)
print('切比雪夫级数逼近多项式：')
print(csa.approximation_poly)
print('切比雪夫级数逼近的最大绝对误差是：')
print(csa.max_abs_error)
csa.plt_approximation()
```
