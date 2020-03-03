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
#matplotlib.use('tkAgg')
#import matplotlib.pyplot as plt
#from matplotlib.ticker import NullFormatter
from scipy.stats import chi2
import GPy


###########################################################################################################
def run_fats(jd, mag, err):
    """
    function tu run fats and return the features in an array
    """

    #list with the features to be calculated
    feature_list = [
        'CAR_sigma',
        'CAR_mean',
        'CAR_tau',
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

def final_DWR_lc(seed, time_range, dtime, tau, SFinf, xmean, errmag, burn, sampling,timesamp):

    tbase=36500
    dtbase=1
    t_drwbase=np.arange(0,tbase,dtbase)

    #np.random.seed(int(seed))
    np.random.seed(int((time.clock()+seed)))

    y = generateDRW(t_drwbase, 0.0, tau, SFinf, xmean)
    ysig = np.zeros_like(t_drwbase)
    #y_obs = y + np.random.normal(0., np.random.normal(magerr, 0.005), len(t_drwbase)) # Heteroskedastic
    #ysig_obs = np.ones_like(t_drwbase) * magerr
    ysig_obs=np.abs(np.random.normal(errmag, 0.004, len(y)))#np.ones(len(y))*errmag
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
    parameters using turbofats, CAR and IAR
    """

    #t,y,ysig,y_obs,ysig_obs = lc.gen_DRW_long(seed,time_range,dtime,mag,errmag,tau,SFinf,sampling,timesamp)
    t,y,ysig,y_obs,ysig_obs = final_DWR_lc(seed, time_range, dtime, tau, SFinf,mag, errmag, 10000, sampling,timesamp)


    nepochs = len(t)
    trange = t[-1] - t[0]

    #variability parameters
    print(seed, tau, SFinf, " calculating var parameters")
    if sampling: print("QUEST sampling")
    pvar,exvar,err_exvar = var_parameters(t,y,ysig)
    pvar_obs,exvar_obs,err_exvar_obs = var_parameters(t,y_obs,ysig_obs)

    #runnung GP DRW
    print(seed, tau, SFinf, " calculating GP DRW")
    var_gp_obs, tau_gp_obs = OU(t, y_obs, ysig_obs)
    var_gp, tau_gp = OU(t, y, ysig)

    #running IAR
    print(seed, tau, SFinf, " calculating IAR")

    yobs1=(y_obs-np.mean(y_obs))/np.sqrt(np.var(y_obs,ddof=1))
    phi_obs=iar.IAR_loglik(yobs1,t,delta=ysig_obs/np.sqrt(np.var(y_obs,ddof=1)),standarized=True)
    tau_iar_obs = -1.0/np.log(phi_obs)

    y1=(y-np.mean(y))/np.sqrt(np.var(y,ddof=1))
    phi=iar.IAR_loglik(y1,t,delta=ysig/np.sqrt(np.var(y,ddof=1)),standarized=True)
    tau_iar = -1.0/np.log(phi)

    #phi_obs = iar.IAR_loglik((y_obs-np.mean(y_obs))/np.std(y_obs),t,ysig_obs,standarized=True)
    #phi = iar.IAR_loglik((y-np.mean(y))/np.std(y),t,ysig,standarized=True)


    #print("IAR results: ")
    #print("Tau_IAR = ", tau_iar)
    #print("phi_IAR = ", phi)
    #print("mu_IAR = ", mu)
    #print("sigma_IAR = ", sigma)

    #running CAR (Pichara)
    print(seed, tau, SFinf, " calculating CAR")
    tau_car, sigma_car = car.calculateCAR(t,y,ysig)
    tau_car_obs, sigma_car_obs = car.calculateCAR(t,y_obs,ysig_obs)

    #print("CAR results: ")
    #print("Tau_car = ", tau_car)
    #print("sigma_car = ", sigma_car)


    #running fats
    print(seed, tau, SFinf, " calculating FATS")
    results_fats = run_fats(t, y_obs, ysig_obs)
    tau_fats_obs = results_fats['CAR_tau']
    mu_fats_obs = results_fats['CAR_tau']*results_fats['CAR_mean']
    sigma_fats_obs = results_fats['CAR_sigma']

    results_fats = run_fats(t, y, ysig)
    tau_fats = results_fats['CAR_tau']
    mu_fats = results_fats['CAR_tau']*results_fats['CAR_mean']
    sigma_fats = results_fats['CAR_sigma']

    #print("FATS results: ")
    #print("Tau_fats = ", tau_fats)
    #print("mean_fats = ", mu_fats)
    #print("sigma_fats = ", sigma_fats)



    #return(tau_iar,phi,tau_fats,mu_fats,sigma_fats)
    return(seed,nepochs,trange,tau, SFinf,pvar,exvar,err_exvar,pvar_obs,exvar_obs,err_exvar_obs, var_gp, tau_gp, var_gp_obs, tau_gp_obs,tau_iar,phi,tau_iar_obs,phi_obs,tau_car,sigma_car,tau_car_obs,sigma_car_obs,tau_fats,sigma_fats,tau_fats_obs,sigma_fats_obs)

###########################################################################################################
#test with irregular sampling

def get_sampling(filename):
    a=pf.open(filename)
    dat=a[1].data
    jd=dat['JD']
    mag=dat['Q']
    err=dat['errQ']

    return((jd-jd[0]),np.mean(mag),np.mean(err))


samp_file_short='bin3_onechip_30.595546_-1.961014_XMM_LSS.fits'
samp_file_med='bin3_morechip_151.393520_2.171140_COSMOS.fits'
samp_file_long='bin3_onechip_32.912634_-4.435199_XMM_LSS.fits'

tobs_short,mag_short,err_short=get_sampling(samp_file_short)
tobs_med,mag_med,err_med=get_sampling(samp_file_med)

tobs_long,mag_long,err_long=get_sampling(samp_file_long)

tobs_super_long=np.concatenate((tobs_long,tobs_long+tobs_long[-1]+200))

tobs_hyper_long=np.concatenate((tobs_super_long,tobs_super_long+tobs_super_long[-1]+300))

tobs_mega_long=np.concatenate((tobs_hyper_long,tobs_hyper_long+tobs_hyper_long[-1]+300))

###########################################################################################################
#running the simulations in multiple cores

def multi_run_wrapper(args):
#function necessary for the use of pool.map with different arguments
   return sim_drw(*args)


def run_sim(nsim,ncores,time_range , dtime, mag, errmag, tau, SFinf, sampling, timesamp,save_file):

    arg_list=[]
    #the array with the arguments is generated, this is necessary to use pool.map
    for i in range(0,nsim):
        arg_list.append((i,time_range,dtime,mag,errmag,tau,SFinf,sampling,timesamp))

    pool = Pool(processes=ncores)
    results = pool.map(multi_run_wrapper,arg_list)
    pool.close()
    pool.join()

    head='seed  num_epochs  time_range_out  tau_org  SFinf_org  pvar  exvar  err_exvar  pvar_obs  exvar_obs  err_exvar_obs  var_gp  tau_gp  var_gp_obs  tau_gp_obs  tau_iar  phi_iar  tau_iar_obs  phi_iar_obs  tau_car  sigma_car  tau_car_obs  sigma_car_obs  tau_fats  sigma_fats  tau_fats_obs  sigma_fats_obs'

    np.savetxt(save_file,results,header=head)
    return (results)

###########################################################################################################
#running simulations with regular sampling

results = run_sim(1000,8,1700, 5, 19.0, 0.02, 5, 0.2, False, 0.0,'sim_results_Jun07/DRW_reg_samp_dl1700_dt_5_tau5_SFinf02.txt')
del results

results = run_sim(1000,8,1700, 5, 19.0, 0.02, 10, 0.2, False, 0.0,'sim_results_Jun07/DRW_reg_samp_dl1700_dt_5_tau10_SFinf02.txt')
del results

results = run_sim(1000,8,1700, 5, 19.0, 0.02, 20, 0.2, False, 0.0,'sim_results_Jun07/DRW_reg_samp_dl1700_dt_5_tau20_SFinf02.txt')
del results

results = run_sim(1000,8,1700, 5, 19.0, 0.02, 35, 0.2, False, 0.0,'sim_results_Jun07/DRW_reg_samp_dl1700_dt_5_tau35_SFinf02.txt')
del results

results = run_sim(1000,8,1700, 5, 19.0, 0.02, 50, 0.2, False, 0.0,'sim_results_Jun07/DRW_reg_samp_dl1700_dt_5_tau50_SFinf02.txt')
del results

results = run_sim(1000,8,1700, 5, 19.0, 0.02, 75, 0.2, False, 0.0,'sim_results_Jun07/DRW_reg_samp_dl1700_dt_5_tau75_SFinf02.txt')
del results

results = run_sim(1000,8,1700, 5, 19.0, 0.02, 100, 0.2, False, 0.0,'sim_results_Jun07/DRW_reg_samp_dl1700_dt_5_tau100_SFinf02.txt')
del results

results = run_sim(1000,8,1700, 5, 19.0, 0.02, 125, 0.2, False, 0.0,'sim_results_Jun07/DRW_reg_samp_dl1700_dt_5_tau125_SFinf02.txt')
del results

results = run_sim(1000,8,1700, 5, 19.0, 0.02, 150, 0.2, False, 0.0,'sim_results_Jun07/DRW_reg_samp_dl1700_dt_5_tau150_SFinf02.txt')
del results

results = run_sim(1000,8,1700, 5, 19.0, 0.02, 200, 0.2, False, 0.0,'sim_results_Jun07/DRW_reg_samp_dl1700_dt_5_tau200_SFinf02.txt')
del results

results = run_sim(1000,8,1700, 5, 19.0, 0.02, 225, 0.2, False, 0.0,'sim_results_Jun07/DRW_reg_samp_dl1700_dt_5_tau225_SFinf02.txt')
del results

results = run_sim(1000,8,1700, 5, 19.0, 0.02, 250, 0.2, False, 0.0,'sim_results_Jun07/DRW_reg_samp_dl1700_dt_5_tau250_SFinf02.txt')
del results

results = run_sim(1000,8,1700, 5, 19.0, 0.02, 275, 0.2, False, 0.0,'sim_results_Jun07/DRW_reg_samp_dl1700_dt_5_tau275_SFinf02.txt')
del results

results = run_sim(1000,8,1700, 5, 19.0, 0.02, 300, 0.2, False, 0.0,'sim_results_Jun07/DRW_reg_samp_dl1700_dt_5_tau300_SFinf02.txt')
del results

results = run_sim(1000,8,1700, 5, 19.0, 0.02, 350, 0.2, False, 0.0,'sim_results_Jun07/DRW_reg_samp_dl1700_dt_5_tau350_SFinf02.txt')
del results

results = run_sim(1000,8,1700, 5, 19.0, 0.02, 400, 0.2, False, 0.0,'sim_results_Jun07/DRW_reg_samp_dl1700_dt_5_tau400_SFinf02.txt')
del results

results = run_sim(1000,8,1700, 5, 19.0, 0.02, 500, 0.2, False, 0.0,'sim_results_Jun07/DRW_reg_samp_dl1700_dt_5_tau500_SFinf02.txt')
del results

results = run_sim(1000,8,1700, 5, 19.0, 0.02, 750, 0.2, False, 0.0,'sim_results_Jun07/DRW_reg_samp_dl1700_dt_5_tau750_SFinf02.txt')
del results

results = run_sim(1000,8,1700, 5, 19.0, 0.02, 1000, 0.2, False, 0.0,'sim_results_Jun07/DRW_reg_samp_dl1700_dt_5_tau1000_SFinf02.txt')
del results

results = run_sim(1000,8,1700, 5, 19.0, 0.02, 1250, 0.2, False, 0.0,'sim_results_Jun07/DRW_reg_samp_dl1700_dt_5_tau1250_SFinf02.txt')
del results

results = run_sim(1000,8,1700, 5, 19.0, 0.02, 1500, 0.2, False, 0.0,'sim_results_Jun07/DRW_reg_samp_dl1700_dt_5_tau1500_SFinf02.txt')
del results

###########################################################################################################
#running simulations with quest sampling

results = run_sim(1000,8,1700, 5, 19.0, 0.02, 5, 0.2, True, tobs_long,'sim_results_Jun07/DRW_quest_samp_tau5_SFinf02.txt')
del results

results = run_sim(1000,8,1700, 5, 19.0, 0.02, 10, 0.2, True, tobs_long,'sim_results_Jun07/DRW_quest_samp_tau10_SFinf02.txt')
del results

results = run_sim(1000,8,1700, 5, 19.0, 0.02, 20, 0.2, True, tobs_long,'sim_results_Jun07/DRW_quest_samp_tau20_SFinf02.txt')
del results

results = run_sim(1000,8,1700, 5, 19.0, 0.02, 35, 0.2, True, tobs_long,'sim_results_Jun07/DRW_quest_samp_tau35_SFinf02.txt')
del results

results = run_sim(1000,8,1700, 5, 19.0, 0.02, 50, 0.2, True, tobs_long,'sim_results_Jun07/DRW_quest_samp_tau50_SFinf02.txt')
del results

results = run_sim(1000,8,1700, 5, 19.0, 0.02, 75, 0.2, True, tobs_long,'sim_results_Jun07/DRW_quest_samp_tau75_SFinf02.txt')
del results

results = run_sim(1000,8,1700, 5, 19.0, 0.02, 100, 0.2, True, tobs_long,'sim_results_Jun07/DRW_quest_samp_tau100_SFinf02.txt')
del results

results = run_sim(1000,8,1700, 5, 19.0, 0.02, 125, 0.2, True, tobs_long,'sim_results_Jun07/DRW_quest_samp_tau125_SFinf02.txt')
del results

results = run_sim(1000,8,1700, 5, 19.0, 0.02, 150, 0.2, True, tobs_long,'sim_results_Jun07/DRW_quest_samp_tau150_SFinf02.txt')
del results

results = run_sim(1000,8,1700, 5, 19.0, 0.02, 200, 0.2, True, tobs_long,'sim_results_Jun07/DRW_quest_samp_tau200_SFinf02.txt')
del results

results = run_sim(1000,8,1700, 5, 19.0, 0.02, 225, 0.2, True, tobs_long,'sim_results_Jun07/DRW_quest_samp_tau225_SFinf02.txt')
del results

results = run_sim(1000,8,1700, 5, 19.0, 0.02, 250, 0.2, True, tobs_long,'sim_results_Jun07/DRW_quest_samp_tau250_SFinf02.txt')
del results

results = run_sim(1000,8,1700, 5, 19.0, 0.02, 275, 0.2, True, tobs_long,'sim_results_Jun07/DRW_quest_samp_tau275_SFinf02.txt')
del results

results = run_sim(1000,8,1700, 5, 19.0, 0.02, 300, 0.2, True, tobs_long,'sim_results_Jun07/DRW_quest_samp_tau300_SFinf02.txt')
del results

results = run_sim(1000,8,1700, 5, 19.0, 0.02, 350, 0.2, True, tobs_long,'sim_results_Jun07/DRW_quest_samp_tau350_SFinf02.txt')
del results

results = run_sim(1000,8,1700, 5, 19.0, 0.02, 400, 0.2, True, tobs_long,'sim_results_Jun07/DRW_quest_samp_tau400_SFinf02.txt')
del results

results = run_sim(1000,8,1700, 5, 19.0, 0.02, 500, 0.2, True, tobs_long,'sim_results_Jun07/DRW_quest_samp_tau500_SFinf02.txt')
del results

results = run_sim(1000,8,1700, 5, 19.0, 0.02, 750, 0.2, True, tobs_long,'sim_results_Jun07/DRW_quest_samp_tau750_SFinf02.txt')
del results

results = run_sim(1000,8,1700, 5, 19.0, 0.02, 1000, 0.2, True, tobs_long,'sim_results_Jun07/DRW_quest_samp_tau1000_SFinf02.txt')
del results

results = run_sim(1000,8,1700, 5, 19.0, 0.02, 1250, 0.2, True, tobs_long,'sim_results_Jun07/DRW_quest_samp_tau1250_SFinf02.txt')
del results

results = run_sim(1000,8,1700, 5, 19.0, 0.02, 1500, 0.2, True, tobs_long,'sim_results_Jun07/DRW_quest_samp_tau1500_SFinf02.txt')
del results


###########################################################################################################
#running simulations with regular sampling and fixed tau:

results = run_sim(1000,8,1000, 5, 19.0, 0.02, 300, 0.2, False, 0.0,'sim_results_Jun07/DRW_reg_samp_dl1000_dt_5_tau300_SFinf02.txt')
del results

results = run_sim(1000,8,500, 5, 19.0, 0.02, 300, 0.2, False, 0.0,'sim_results_Jun07/DRW_reg_samp_dl500_dt_5_tau300_SFinf02.txt')
del results

results = run_sim(1000,8,1500, 5, 19.0, 0.02, 300, 0.2, False, 0.0,'sim_results_Jun07/DRW_reg_samp_dl1500_dt_5_tau300_SFinf02.txt')
del results

results = run_sim(1000,8,6000, 5, 19.0, 0.02, 300, 0.2, False, 0.0,'sim_results_Jun07/DRW_reg_samp_dl6000_dt_5_tau300_SFinf02.txt')
del results

results = run_sim(1000,8,3000, 5, 19.0, 0.02, 300, 0.2, False, 0.0,'sim_results_Jun07/DRW_reg_samp_dl3000_dt_5_tau300_SFinf02.txt')
del results

results = run_sim(1000,8,4000, 5, 19.0, 0.02, 300, 0.2, False, 0.0,'sim_results_Jun07/DRW_reg_samp_dl4000_dt_5_tau300_SFinf02.txt')
del results

results = run_sim(1000,8,5000, 5, 19.0, 0.02, 300, 0.2, False, 0.0,'sim_results_Jun07/DRW_reg_samp_dl5000_dt_5_tau300_SFinf02.txt')
del results


###########################################################################################################
#running simulations with irregular sampling and fixed tau:

results = run_sim(1000,8,2000, 5, 19.0, 0.02, 300, 0.2, True, tobs_super_long,'sim_results_Jun07/DRW_quest_samp_dlsuper_tau300_SFinf02.txt')
del results

results = run_sim(1000,8,2000, 5, 19.0, 0.02, 300, 0.2, True, tobs_hyper_long,'sim_results_Jun07/DRW_quest_samp_dlhyper_tau300_SFinf02.txt')
del results

results = run_sim(1000,8,2000, 5, 19.0, 0.02, 300, 0.2, True, tobs_mega_long,'sim_results_Jun07/DRW_quest_samp_dlmega_tau300_SFinf02.txt')
del results

results = run_sim(1000,8,2000, 5, 19.0, 0.02, 300, 0.2, True, tobs_long,'sim_results_Jun07/DRW_quest_samp_dllong_tau300_SFinf02.txt')
del results

results = run_sim(1000,8,2000, 5, 19.0, 0.02, 300, 0.2, True, tobs_short,'sim_results_Jun07/DRW_quest_samp_dlshort_tau300_SFinf02.txt')
del results

results = run_sim(1000,8,2000, 5, 19.0, 0.02, 300, 0.2, True, tobs_med,'sim_results_Jun07/DRW_quest_samp_dlmed_tau300_SFinf02.txt')
del results




#results = sim_drw(100, 2000 , 5, 19.0, 0.02, 400, 0.2, True, tobs_long)
#print(results)
