# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 16:32:22 2022

@author: cosbo
"""

import opt
import numpy as np
import json
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import quad



def integrand(z, om):
    return ((1 - om) * (1+z)**3 + om)**(-1/2)


def dintegrand(z, om):
    """
    производная подинтегрального выражения по dOmega
    """
    numerator = (1 + z)**3 - 1
    denominator = 2 * (om - (om - 1) * (z + 1)**3)**(3/2)
    return numerator / denominator


def mu_of_z(z, args):
    """
    буквально функция расстояния от красного смещения
    """
    om = args[0]
    h0 = args[1]
    d = [3e11 * (1 + zz) * quad(integrand, 0, zz, om)[0] / h0 for zz in z]
    return 5*np.log10(d) - 5


def dmu_dz(z, args):
    """
    якобиан
    """
    om = args[0]
    h0 = args[1]
    dh0 = np.asarray([-5/(np.log(10)*h0) for zz in z])
    dom = np.asarray([5*quad(dintegrand, 0, zz, om)[0] /
                      (quad(integrand, 0, zz, om)[0] * np.log(10))
                      for zz in z])
    return np.transpose(np.vstack((dom, dh0)))


if __name__ == '__main__':

    data = []
    with open('jla_mub.txt') as f:
        for i, line in enumerate(f):
            if i != 0:
                data.append(line.split(" "))
    data = np.transpose(np.asarray(data, dtype=float))

    # приближение
    r_gauss = opt.gauss_newton(data, mu_of_z, dmu_dz, [0.5, 50], k=0.9)
    r_lm = opt.lm(data, mu_of_z, dmu_dz, [0.5, 50])

    # красивый график
    style_default = matplotlib.font_manager.FontProperties()
    style_default.set_size('x-large')
    style_default.set_family(['Calibri', 'Helvetica', 'Arial', 'serif'])
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    # если я всё правильно помню, обе оси в безразмерных величинах
    # если вдруг нет, не бейте, это по астрономии пробел, а не по питону
    ax.set_xlabel('z, красное смещение', fontproperties=style_default)
    ax.set_ylabel('mu, модуль расстояния', fontproperties=style_default)
    datapoints, = plt.plot(data[0], data[1], 'o', markersize=3)
    xx = np.linspace(data[0, 0], data[0, -1], 100)
    function, = plt.plot(xx, mu_of_z(xx, r_lm.x), lw=2)
    datapoints.set_label('данные')
    function.set_label('модельная кривая')
    plt.legend(loc='best', prop=style_default)
    plt.savefig('mu-z.png')
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(r_gauss.cost)
    plt.plot(r_lm.cost)
    ax.set_yscale('log')