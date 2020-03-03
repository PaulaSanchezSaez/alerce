import os
import numpy as np
import pandas as pd
import turbofats as FATS
import statsmodels.api as sm
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from scipy.interpolate import splrep, splev





def read_lc(lc_name,filter_id):

    print("########## reading lc %s  ##########" % (lc_name))

    lc = pd.read_csv(lc_name)
    #lc = lc.dropna()
    lc = lc[lc['fid']==filter_id]
    lc = lc[(lc['sigmapsf_corr']<0.5) & (np.abs(lc['magpsf_corr'])<30)]
    jd = lc['mjd'].values
    mag = lc['magpsf_corr'].values
    errmag = lc['sigmapsf_corr'].values
    sgscore1 = np.median(lc['sgscore1'].values)
    ra = np.mean(lc['ra'].values)
    dec = np.mean(lc['dec'].values)

    return(ra,dec,sgscore1,jd,mag,errmag)


################################################################################
def moving_average_ts(jd,mag,window):
    mov_av = []
    days = jd
    for day in days:
        mag_win = mag[np.where((jd>(day-window*0.5)) & (jd<(day+window*0.5)))]
        aux = np.mean(mag_win)
        mov_av.append(aux)

    mov_av = np.array(mov_av)
    return(mov_av)


################################################################################

import numpy as np
import time
import math


def tricubic(x):
    y = np.zeros_like(x)
    idx = (x >= -1) & (x <= 1)
    y[idx] = np.power(1.0 - np.power(np.abs(x[idx]), 3), 3)
    return y


class Loess(object):

    @staticmethod
    def normalize_array(array):
        min_val = np.min(array)
        max_val = np.max(array)
        return (array - min_val) / (max_val - min_val), min_val, max_val

    def __init__(self, xx, yy, degree=1):
        self.n_xx, self.min_xx, self.max_xx = self.normalize_array(xx)
        self.n_yy, self.min_yy, self.max_yy = self.normalize_array(yy)
        self.degree = degree

    @staticmethod
    def get_min_range(distances, window):
        min_idx = np.argmin(distances)
        n = len(distances)
        if min_idx == 0:
            return np.arange(0, window)
        if min_idx == n-1:
            return np.arange(n - window, n)

        min_range = [min_idx]
        while len(min_range) < window:
            i0 = min_range[0]
            i1 = min_range[-1]
            if i0 == 0:
                min_range.append(i1 + 1)
            elif i1 == n-1:
                min_range.insert(0, i0 - 1)
            elif distances[i0-1] < distances[i1+1]:
                min_range.insert(0, i0 - 1)
            else:
                min_range.append(i1 + 1)
        return np.array(min_range)

    @staticmethod
    def get_weights(distances, min_range):
        max_distance = np.max(distances[min_range])
        weights = tricubic(distances[min_range] / max_distance)
        return weights

    def normalize_x(self, value):
        return (value - self.min_xx) / (self.max_xx - self.min_xx)

    def denormalize_y(self, value):
        return value * (self.max_yy - self.min_yy) + self.min_yy

    def estimate(self, x, window, use_matrix=False, degree=1):
        n_x = self.normalize_x(x)
        distances = np.abs(self.n_xx - n_x)
        min_range = self.get_min_range(distances, window)
        weights = self.get_weights(distances, min_range)

        if use_matrix or degree > 1:
            wm = np.multiply(np.eye(window), weights)
            xm = np.ones((window, degree + 1))

            xp = np.array([[math.pow(n_x, p)] for p in range(degree + 1)])
            for i in range(1, degree + 1):
                xm[:, i] = np.power(self.n_xx[min_range], i)

            ym = self.n_yy[min_range]
            xmt_wm = np.transpose(xm) @ wm
            beta = np.linalg.pinv(xmt_wm @ xm) @ xmt_wm @ ym
            y = (beta @ xp)[0]
        else:
            xx = self.n_xx[min_range]
            yy = self.n_yy[min_range]
            sum_weight = np.sum(weights)
            sum_weight_x = np.dot(xx, weights)
            sum_weight_y = np.dot(yy, weights)
            sum_weight_x2 = np.dot(np.multiply(xx, xx), weights)
            sum_weight_xy = np.dot(np.multiply(xx, yy), weights)

            mean_x = sum_weight_x / sum_weight
            mean_y = sum_weight_y / sum_weight

            b = (sum_weight_xy - mean_x * mean_y * sum_weight) / \
                (sum_weight_x2 - mean_x * mean_x * sum_weight)
            a = mean_y - b * mean_x
            y = a + b * n_x
        return self.denormalize_y(y)

'''
def main():
    xx = np.array([0.5578196, 2.0217271, 2.5773252, 3.4140288, 4.3014084,
                   4.7448394, 5.1073781, 6.5411662, 6.7216176, 7.2600583,
                   8.1335874, 9.1224379, 11.9296663, 12.3797674, 13.2728619,
                   4.2767453, 15.3731026, 15.6476637, 18.5605355, 18.5866354,
                   18.7572812])
    yy = np.array([18.63654, 103.49646, 150.35391, 190.51031, 208.70115,
                   213.71135, 228.49353, 233.55387, 234.55054, 223.89225,
                   227.68339, 223.91982, 168.01999, 164.95750, 152.61107,
                   160.78742, 168.55567, 152.42658, 221.70702, 222.69040,
                   243.18828])

    loess = Loess(xx, yy)

    for x in xx:
        y = loess.estimate(x, window=7, use_matrix=False, degree=1)
        print(x, y)
'''
################################################################################


#ra,dec,sgscore1,jd,mag,errmag = read_lc('ZTF17aaaivav_AGN_gband.csv',1)
ra,dec,sgscore1,jd,mag,errmag = read_lc('ZTF18aagteoy_blazar_gband.csv',1)

'''

lowess = sm.nonparametric.lowess
z = lowess(mag, jd,frac=0.25)

plt.plot(jd,mag,'ro')
plt.title('lowess')
plt.plot(z[:,0],z[:,1])
plt.show()



rms =(np.mean((mag-z[:,1])**2))/(np.mean(mag)**2)
print("RMS = ",rms)


loess = Loess(jd,mag)

y=[]
for x in jd:
    y.append(loess.estimate(x, window=int(len(mag)*0.3), degree=3))

x= jd
y=np.array(y)


plt.plot(jd,mag,'ro')
plt.title('Loess 2')
plt.plot(x,y)
plt.show()

rms =(np.mean((mag-y)**2))/(np.mean(mag)**2)
print("RMS = ",rms)



yhat = savgol_filter(mag, 21, 2)

rms =(np.mean((mag-yhat)**2))/(np.mean(mag)**2)
print("RMS = ",rms)

plt.plot(jd,mag,'ro')
plt.title('savgol_filter')
plt.plot(jd,yhat)
plt.show()

'''

ss = (len(mag)+np.sqrt(2*len(mag)))/2
print(ss)
bspl = splrep(jd,mag,w=1/errmag,k=3,s=100)
bspl_y = splev(jd,bspl)

rms =(np.mean((mag-bspl_y)**2))/(np.mean(mag)**2)
print("RMS = ",rms)

plt.plot(jd,mag,'ro')
plt.title('splrep')
plt.plot(jd,bspl_y)
plt.show()


mov_av = moving_average_ts(jd,mag,30)

#rms =(np.mean((mag-mov_av)**2))/(np.mean(mag)**2)
rms = np.sqrt(np.mean((mag-mov_av)**2))
print("RMS = ",rms)

plt.plot(jd,mag,'ro')
plt.title('moving_average')
plt.plot(jd,mov_av)
plt.show()
