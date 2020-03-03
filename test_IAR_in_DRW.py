import numpy as np
import astropy.io.fits as pf
from astroML.time_series import lomb_scargle, generate_damped_RW
from astroML.time_series import ACF_scargle, ACF_EK
import lc_simulation as lc
from multiprocessing import Pool
import time
import FunctionsIATS as iar
import CAR as car
import turbofats as FATS
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import GPy
import var_features_June2019 as vf
from scipy.stats import chi2
import var_features_June2019 as vf

###########################################################################################################
def run_fats(jd, mag, err):
    """
    function tu run fats and return the features in an array
    """

    #list with the features to be calculated
    feature_list = [
        'Mean',
        #'Std',
        #'Meanvariance',
        #'MedianBRP',
        'Rcs',
        #'PeriodLS',
        #'Period_fit',
        #'Color',
        #'Autocor_length',
        #'SlottedA_length',
        'StetsonK',
        #'StetsonK_AC',
        #'Eta_e',
        #'Amplitude',
        #'PercentAmplitude',
        #'Con',
        #'LinearTrend',
        #'Beyond1Std',
        #'FluxPercentileRatioMid20',
        #'FluxPercentileRatioMid35',
        #'FluxPercentileRatioMid50',
        #'FluxPercentileRatioMid65',
        #'FluxPercentileRatioMid80',
        #'PercentDifferenceFluxPercentile',
        #'Q31',
        'ExcessVar',
        'Pvar',
        'CAR_sigma',
        'CAR_mean',
        'CAR_tau',
        'SF_ML_amplitude',
        'SF_ML_gamma',
        'GP_DRW_sigma',
        'GP_DRW_tau',
    ]

    data_array = np.array([mag, jd, err])
    data_ids = ['magnitude', 'time', 'error']
    feat_space = FATS.FeatureSpace(featureList=feature_list, Data=data_ids)
    feat_vals = feat_space.calculateFeature(data_array).result(method='dict')
    #f_results = feat_vals.result(method='array')
    #f_features = feat_vals.result(method='features')

    return (feat_vals)

###########################################################################################################
#to get P, exvar y exvar_err

def var_parameters(jd,mag,err):
#function to calculate the probability of a light curve to be variable, and the excess variance

    mean=np.mean(mag)
    nepochs=float(len(jd))

    chi= np.sum( (mag - mean)**2. / err**2. )
    q_chi=chi2.cdf(chi,(nepochs-1))


    a=(mag-mean)**2
    ex_var=(np.sum(a-err**2)/((nepochs*(mean**2))))
    sd=np.sqrt((1./(nepochs-1))*np.sum(((a-err**2)-ex_var*(mean**2))**2))
    ex_verr=sd/((mean**2)*np.sqrt(nepochs))


    return [q_chi,ex_var,ex_verr]


###########################################################################################################

def getDRWMag(tau, SFinf, mag, dt):
  loc = mag * np.exp(-dt / tau)
  scale = SFinf * np.sqrt((1. - np.exp(-2. * dt / tau)) / 2.)
  return loc + np.random.normal(0., scale)


def generateDRW(t, z, tau, SFinf = 0.3, xmean = 0, burn = 10000):
  # Generate in rest frame
  n = len(t)
  t_rest = t / (1. + z)
  tau /= (1. + z)
  dt = np.diff(t_rest)
  mag = np.zeros(n)

  mag[0] = np.random.normal(0, SFinf / np.sqrt(2.))
  for i in range(burn):
    mag[0] = getDRWMag(tau, SFinf, mag[0], dt[np.random.randint(n - 1)])
  for i in range(n - 1):
    mag[i + 1] = getDRWMag(tau, SFinf, mag[i], dt[i])
  return xmean + mag

def final_DWR_lc(seed, time_range, dtime, tau, SFinf, xmean, magerr, burn, sampling,timesamp):

    tbase=36500
    dtbase=1
    t_drwbase=np.arange(0,tbase,dtbase)

    np.random.seed(int(seed))

    y = generateDRW(t_drwbase, 0.0, tau, SFinf, xmean)
    ysig = np.zeros_like(t_drwbase)
    #y_obs = y + np.random.normal(0., np.random.normal(magerr, 0.005), len(t_drwbase)) # Heteroskedastic
    #ysig_obs = np.ones_like(t_drwbase) * magerr
    ysig_obs=np.abs(np.random.normal(magerr, 0.004, len(y)))#np.ones(len(y))*errmag
    y_obs = y+np.random.normal(0., ysig_obs)  #np.random.normal(y, errmag)

    if sampling:

        timesamp=np.round(timesamp,decimals=0).astype(np.int)
        tstar = np.random.randint(14600,tbase-np.int((timesamp[-1]-timesamp[0])),size=1)
        t_drw = tstar+((timesamp-timesamp[0]))
        #print(t_drw)

        y=y[t_drw]
        ysig=ysig[t_drw]
        ysig_obs=ysig_obs[t_drw]
        y_obs=y_obs[t_drw]

        t_drw=t_drw-t_drw[0]


    else:

        tstar = np.random.randint(14600,tbase-time_range,size=1)
        tend = tstar+time_range
        t_drw=np.arange(tstar,tend,dtime)
        y=y[t_drw]
        ysig=ysig[t_drw]
        ysig_obs=ysig_obs[t_drw]
        y_obs=y_obs[t_drw]

        t_drw=t_drw-t_drw[0]

    return (t_drw,y,ysig,y_obs,ysig_obs)


###########################################################################################################
# Mathew's method to model DRW with gaussian process

def OU(t, mag, err):
  # Fit GP OU model
  mag -= mag.mean()
  kern = GPy.kern.OU(1)
  m = GPy.models.GPHeteroscedasticRegression(t[:, None], mag[:, None], kern)
  m['.*het_Gauss.variance'] = abs(err ** 2.)[:, None] # Set the noise parameters to the error in Y
  m.het_Gauss.variance.fix() # We can fix the noise term, since we already know it
  m.optimize()
  pars = [m.OU.variance.values[0], m.OU.lengthscale.values[0]] # sigma^2, tau
  return pars[0], pars[1]


###########################################################################################################

def sim_drw(seed, time_range , dtime, mag, errmag, tau, SFinf, sampling, timesamp):
    """
    Function to simulate a DRW process and estimate its tau, sigma, and mean
    parameters using turbofats and IAR
    """

    #t,y,ysig,y_obs,ysig_obs = lc.gen_DRW_long(seed,time_range,dtime,mag,errmag,tau,SFinf,sampling,timesamp)
    t,y,ysig,y_obs,ysig_obs = final_DWR_lc(seed, time_range, dtime, tau, SFinf,mag, errmag, 10000, sampling,timesamp)
    #t,y,ysig,y_obs,ysig_obs = np.genfromtxt('test_lc.txt',comments="#",unpack=True)
    #np.savetxt('test_lc.txt',np.array([t,y,ysig,y_obs,ysig_obs]).transpose(),header='t y ysig y_obs ysig_obs')

    plt.errorbar(t,y_obs,ysig_obs,color='red',ecolor='red',marker='s',linestyle='None',label='obs',zorder=0)
    plt.plot(t,y,'b.',label='org',zorder=10)
    plt.legend()
    plt.xlabel('days')
    plt.ylabel('magnitude')
    #plt.ylim(18.3,19.3)
    #plt.xlim(-20,1800)
    plt.gca().invert_yaxis()
    plt.show()

    print(len(y_obs))
    print("mag mean = ",np.mean(y_obs))

    pvar_obs,exvar_obs,err_exvar_obs = vf.var_parameters(t,y_obs,ysig_obs)
    print("var feat results: ")
    print("pvar_obs = ", pvar_obs)
    print("exvar_obs = ", exvar_obs)
    print("mag mean = ",np.mean(y_obs))

    print(seed, " calculating ML SF")
    gamma_ML_obs,A_ML_obs = vf.SF_ML(t,y_obs,ysig_obs)
    print(gamma_ML_obs,A_ML_obs)
    print("mag mean = ",np.mean(y_obs))


    #runnung GP DRW
    var_gp, tau_gp = vf.GP_DRW(t, y_obs, ysig_obs)
    print("GP results: ")
    print("Tau_GP = ", tau_gp)
    print("var_GP = ", var_gp)
    print("mag mean = ",np.mean(y_obs))


    #running IAR
    #yobs1=(y_obs-np.mean(y_obs))/np.sqrt(np.var(y_obs,ddof=1))
    #phi=iar.IAR_loglik(yobs1,t,delta=ysig_obs/np.sqrt(np.var(y_obs,ddof=1)),standarized=True)
    #phi = iar.IAR_loglik((y_obs-np.mean(y_obs))/np.std(y_obs),t,ysig_obs,standarized=True)
    #phi = iar.IAR_loglik((y-np.mean(y))/np.std(y),t,standarized=True)

    #tau_iar = -1.0/np.log(phi)
    phi, tau_iar = vf.IAR_kalman(t, y_obs, ysig_obs)

    print("IAR results: ")
    print("Tau_IAR = ", tau_iar)
    print("phi_IAR = ", phi)
    print("mag mean = ",np.mean(y_obs))
    #print("mu_IAR = ", mu)
    #print("sigma_IAR = ", sigma)

    #iar.plot_IAR_phi_loglik((y_obs-np.mean(y_obs))/np.std(y_obs),t,1000)

    tau_car, sigma_car = car.calculateCAR(t,y_obs,ysig_obs)

    print("CAR results: ")
    print("Tau_car = ", tau_car)
    print("sigma_car = ", sigma_car)
    print("mag mean = ",np.mean(y_obs))


    #min_sig_car, min_tau_car = car.plot_CAR_Lik([0.00001,1000],[0.0, 3], t,y,ysig,200)


    #running fats
    print("mag mean = ",np.mean(y_obs))
    results_fats = run_fats(t, y_obs, ysig_obs)
    tau_fats = results_fats['GP_DRW_tau']
    sigma_fats = results_fats['GP_DRW_sigma']
    gamma_fats = results_fats['SF_ML_gamma']
    A_fats = results_fats['SF_ML_amplitude']
    pvar_fats = results_fats['Pvar']
    exvar_fats = results_fats['ExcessVar']
    mean_fats = results_fats['Mean']


    print("mag mean = ",np.mean(y_obs))
    print("FATS results: ")
    print("Tau_fats = ", tau_fats)
    print("sigma_fats = ", sigma_fats)
    print("A_fats = ", A_fats)
    print("gamma_fats = ", gamma_fats)
    print("pvar_fats = ", pvar_fats)
    print("exvar_fats = ", exvar_fats)
    print("mean_fats = ", mean_fats)



    #return(tau_iar,phi,tau_fats,mu_fats,sigma_fats)
    return(tau_iar,phi,tau_car, sigma_car)

###########################################################################################################
#test with regular sampling

#results = sim_drw(191, 1700 , 10, 19.0, 0.02, 400, 0.2, False, 0.0)

###########################################################################################################
#test with irregular sampling

def get_sampling(filename):
    a=pf.open(filename)
    dat=a[1].data
    jd=dat['JD']
    mag=dat['Q']
    err=dat['errQ']

    return((jd-jd[0]),np.mean(mag),np.mean(err))


samp_file_long='bin3_onechip_32.912634_-4.435199_XMM_LSS.fits'

tobs_long,mag_long,err_long=get_sampling(samp_file_long)

results = sim_drw(1909, 1700 , 5, 19.0, 0.02, 20, 0.01, True, tobs_long)
