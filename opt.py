#!/usr/bin/env python3

from collections import namedtuple
import numpy as np

Result = namedtuple('Result', ('nfev', 'cost', 'gradnorm', 'x'))
Result.__doc__ = """Результаты оптимизации

Attributes
----------
nfev : int
    Полное число вызовов модельной функции
cost : 1-d array
    Значения функции потерь 0.5 sum(y - f)^2 на каждом итерационном шаге.
    В случае метода Гаусса—Ньютона длина массива равна nfev, в случае ЛМ-метода
    длина массива — менее nfev
gradnorm : float
    Норма градиента на финальном итерационном шаге
x : 1-d array
    Финальное значение вектора, минимизирующего функцию потерь
"""


def forward_sub(L, b):
    """
    fast solving method for lower-triangular matrix,
    used after Cholesky decomposition

    """
    L = np.atleast_2d(L)
    b = np.atleast_1d(b)
    y = np.zeros_like(b)
    n = b.size
    y[0] = b[0]/L[0, 0]
    for i in range(1, n):
        S = 0
        for j in range(i):
            S += L[i, j]*y[j]
        y[i] = (b[i] - S) / L[i, i]
    return y


def back_sub(U, b):
    """
    fast solving methods for upper-triangular matrix,
    used after Cholesky decompostition

    """
    U = np.atleast_2d(U)
    b = np.atleast_1d(b)
    y = np.zeros_like(b)
    n = b.size
    y[n-1] = b[n-1]/U[n-1, n-1]
    for i in range(n-2, -1, -1):
        S = 0
        for j in range(n-1, i, -1):
            S += U[i, j]*y[j]
        y[i] = (b[i] - S) / U[i, i]
    return y


def lm_step(y, f, j, x0, x, lmbd, nu=2):
    """
    решение уравнения на минимизацию в алгортиме ЛМ при определённых параметрах
    """
    jac_T = j(x, x0).T
    j_tj = jac_T @ j(x, x0)
    lhs = np.asarray(j_tj + lmbd*np.diag(np.diag(j_tj)), dtype=np.float)
    rhs = np.asarray(jac_T @ (y - f(x, x0)), dtype=np.float)
    try:
        L = np.linalg.cholesky(lhs)
    except np.linalg.LinAlgError:
        return x0, np.linalg.norm(y-f(x, x0))
    yy = forward_sub(L, rhs)
    p = back_sub(L.T, yy)
    x0 += p
    res = np.linalg.norm(y-f(x, x0))
    return x0, res


def gauss_newton(data, f, j, x0, k=0.8, tol=1e-3, maxiter=int(100)):
    """
    решение задачи МНК на приближение данных функцией
    все идейные составляющие взяты с википедии

    :param data: массив с измерениями
    :type data: 2d np.array
    :param f: функция от неивестных параметров, возвращающая значения,
    рассчитанные в соответствии с моделью, в виде одномерного массива
    :type f: callable
    :param j: функция от неизвестных параметров, возвращающая якобиан в виде
    двумерного массива
    :type j: callable
    :param x0: массив с начальными приближениями параметров
    :type x0: 1d np.array
    :param k: положительное число меньше единицы, параметр метода,
    defaults to 0.8
    :type k: int, optional
    :param tol:  относительная ошибка, условие сходимости,
    параметр метода, defaults to 1e-3
    :type tol: np.float, optional
    :param maxiter: максимальное, defaults to int(100)
    :type maxiter: int, optional
    :return: смотрите определение Result
    :rtype: namedtuple

    """
    x = data[0]
    y = data[1]
    cost = np.zeros(maxiter+1)  # потеря в памяти, выигрыш в скорости
    i = 0

    while True:
        jac = np.linalg.pinv(j(x, x0))
        x0 = x0 + k*jac@(y - f(x, x0))
        cost[i] = 0.5 * np.linalg.norm(y - f(x, x0))
        if cost[i] <= tol or i >= maxiter:
            break
        i += 1

    fin = np.count_nonzero(cost)
    cost = cost[0:fin]

    return Result(nfev=i, cost=cost, gradnorm=cost[fin-1], x=x0)


def lm(data, f, j, x0, lmbd0=1e-2, nu=2, tol=1e-5, maxiter=int(30)):
    """
    решение задачи МНК на приближение данных функцией
    основные идеи из работы J.J. More, идея смены лямбда-параметра с вики
    (ссылаются на самого Левенберга)

    :param data: массив с измерениями
    :type data: 2d np.array
    :param f: функция от неивестных параметров, возвращающая значения,
    рассчитанные в соответствии с моделью, в виде одномерного массива
    :type f: callable
    :param j: функция от неизвестных параметров, возвращающая якобиан в виде
    двумерного массива
    :type j: callable
    :param x0: массив с начальными приближениями параметров
    :type x0: 1d np.array
    :param lmbd0: положительное число меньше единицы, начальный параметр метода
    defaults to 1e-2
    :type lmbd0: int, optional
    :param tol:  относительная ошибка, условие сходимости,
    параметр метода, defaults to 1e-3
    :type tol: np.float, optional
    :param maxiter: максимальное, defaults to int(30)
    :type maxiter: int, optional
    :return: смотрите определение Result
    :rtype: namedtuple

    """
    x = data[0]
    y = data[1]
    cost = np.zeros(maxiter + 2)
    res0 = np.linalg.norm(y-f(x, x0))
    cost[0] = 0.5 * res0
    i = 0
    while True:
        x0, res = lm_step(y, f, j, x0, x, lmbd0)
        if res0 <= res:
            x0, res = lm_step(y, f, j, x0, x, lmbd0/nu)
            if res0 <= res:
                a = nu
                while res >= res0 and abs(res-res0) >= tol:
                    x0, res = lm_step(y, f, j, x0, x, lmbd0*a)
                    a = a * nu
                    lmbd0 = a
                    if a >= 1e5:
                        print('Warning! Lambda overflow, check the residual')
                        fin = np.count_nonzero(cost)
                        cost = cost[0:fin]
                        return Result(nfev=i, cost=cost,
                                      gradnorm=cost[fin-1], x=x0)
            else:
                lmbd0 = lmbd0/nu
        i += 1
        cost[i] = res
        if res <= tol or abs(cost[i]-cost[i-1]) <= tol or i > maxiter:
            break
        res0 = res

    fin = np.count_nonzero(cost)
    cost = cost[0:fin]
    return Result(nfev=i, cost=cost, gradnorm=cost[fin-1], x=x0)


if __name__ == "__main__":
    pass
