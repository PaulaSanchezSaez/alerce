import numpy as np
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from multiprocessing import Pool
import time
import var_features_June2019 as vf
import lc_simulation as lc
import astropy.io.fits as pf

################################################################################

def SF_sim(seed,time_range,dtime,mag,errmag,model,gamma,sampling,timesamp):

    #generate light curve
    if model=='DRW':
        t,y,ysig,y_obs,ysig_obs = lc.gen_DRW_long(seed,time_range,dtime,mag,errmag,gamma,0.2,sampling,timesamp)
    else:
        alpha = 1.0+2.0*gamma
        t,y,ysig,y_obs,ysig_obs=lc.gen_lc_long(seed,time_range,dtime,mag,errmag,model,alpha,sampling,timesamp)

    nepochs = len(t)
    trange = t[-1] - t[0]

    plt.plot(t,y_obs,'ro')
    plt.show()

    #variability parameters
    print(seed, model, gamma, " calculating var parameters")
    if sampling: print("QUEST sampling")
    pvar,exvar,err_exvar = vf.var_parameters(t,y,ysig)
    pvar_obs,exvar_obs,err_exvar_obs = vf.var_parameters(t,y_obs,ysig_obs)

    #runnung ML SF
    print(seed, model, gamma, " calculating ML SF")
    gamma_ML,A_ML = vf.SF_ML(t,y,ysig)
    gamma_ML_obs,A_ML_obs = vf.SF_ML(t,y_obs,ysig_obs)
    print(gamma_ML_obs,A_ML_obs)

    #runnung Sch10 SF
    print(seed, model, gamma, " calculating Sch10 SF")
    gamma_sch10,A_sch10 = vf.fitSF_Sch10(t,y,ysig,0.1,10,2000)
    gamma_sch10_obs,A_sch10_obs = vf.fitSF_Sch10(t,y_obs,ysig_obs,0.1,10,2000)
    print(gamma_sch10_obs,A_sch10_obs)

    #runnung Bayessian SF
    print(seed, model, gamma, " calculating Bayessian SF")
    gamma_mcmc,A_mcmc = vf.fitSF_mcmc(t,y,ysig)
    gamma_mcmc_obs,A_mcmc_obs = vf.fitSF_mcmc(t,y_obs,ysig_obs)
    #print(gamma_mcmc_obs,A_mcmc_obs)

    return(seed,nepochs,trange,gamma, pvar,exvar,err_exvar,pvar_obs,exvar_obs,err_exvar_obs, gamma_ML,A_ML, gamma_ML_obs,A_ML_obs,gamma_sch10,A_sch10,gamma_sch10_obs,A_sch10_obs,gamma_mcmc[0],A_mcmc[0],gamma_mcmc_obs[0],A_mcmc_obs[0])


################################################################################
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

################################################################################
#running the simulations in multiple cores

def multi_run_wrapper(args):
#function necessary for the use of pool.map with different arguments
   return SF_sim(*args)


def run_sim(nsim,ncores,time_range,dtime,mag,errmag,model,gamma,sampling,timesamp,save_file):

    arg_list=[]
    #the array with the arguments is generated, this is necessary to use pool.map
    for i in range(0,nsim):
        arg_list.append((i,time_range,dtime,mag,errmag,model,gamma,sampling,timesamp))

    pool = Pool(processes=ncores)
    results = pool.map(multi_run_wrapper,arg_list)
    pool.close()
    pool.join()

    head='seed  num_epochs  time_range_out  gamma_input  pvar  exvar  err_exvar  pvar_obs  exvar_obs  err_exvar_obs  gamma_ML  A_ML  gamma_ML_obs  A_ML_obs  gamma_sch10  A_sch10  gamma_sch10_obs  A_sch10_obs  gamma_mcmc  A_mcmc  gamma_mcmc_obs  A_mcmc_obs'

    np.savetxt(save_file,results,header=head)
    return (results)

SF_sim(110,1700,10,19.0,0.02,'power-law',3.0,True, tobs_long)

################################################################################
#running power-law simulations with regular sampling
'''
results0 = run_sim(1000,15,1700,10,19.0,0.02,'power-law',0.0,False,0.0,'sim_results_Jun07/SF_reg_samp_dl1700_dt_10_power_law_gamma0.txt')
del results0

results0 = run_sim(1000,15,1700,10,19.0,0.02,'power-law',0.25,False,0.0,'sim_results_Jun07/SF_reg_samp_dl1700_dt_10_power_law_gamma025.txt')
del results0

results0 = run_sim(1000,15,1700,10,19.0,0.02,'power-law',0.5,False,0.0,'sim_results_Jun07/SF_reg_samp_dl1700_dt_10_power_law_gamma05.txt')
del results0

results0 = run_sim(1000,15,1700,10,19.0,0.02,'power-law',0.75,False,0.0,'sim_results_Jun07/SF_reg_samp_dl1700_dt_10_power_law_gamma075.txt')
del results0

results0 = run_sim(1000,15,1700,10,19.0,0.02,'power-law',1.0,False,0.0,'sim_results_Jun07/SF_reg_samp_dl1700_dt_10_power_law_gamma1.txt')
del results0

results0 = run_sim(1000,15,1700,10,19.0,0.02,'power-law',1.25,False,0.0,'sim_results_Jun07/SF_reg_samp_dl1700_dt_10_power_law_gamma125.txt')
del results0

#results0 = run_sim(1000,15,1700,10,19.0,0.02,'power-law',1.5,False,0.0,'sim_results_Jun07/SF_reg_samp_dl1700_dt_10_power_law_gamma15.txt')
#del results0

################################################################################
#running power-law simulations with QUEST sampling

results0 = run_sim(1000,15,1700,10,19.0,0.02,'power-law',0.0,True, tobs_long,'sim_results_Jun07/SF_quest_samp_power_law_gamma0.txt')
del results0

results0 = run_sim(1000,15,1700,10,19.0,0.02,'power-law',0.25,True, tobs_long,'sim_results_Jun07/SF_quest_samp_power_law_gamma025.txt')
del results0

results0 = run_sim(1000,15,1700,10,19.0,0.02,'power-law',0.5,True, tobs_long,'sim_results_Jun07/SF_quest_samp_power_law_gamma05.txt')
del results0

results0 = run_sim(1000,15,1700,10,19.0,0.02,'power-law',0.75,True, tobs_long,'sim_results_Jun07/SF_quest_samp_power_law_gamma075.txt')
del results0

results0 = run_sim(1000,15,1700,10,19.0,0.02,'power-law',1.0,True, tobs_long,'sim_results_Jun07/SF_quest_samp_power_law_gamma1.txt')
del results0

results0 = run_sim(1000,15,1700,10,19.0,0.02,'power-law',1.25,True, tobs_long,'sim_results_Jun07/SF_quest_samp_power_law_gamma125.txt')
del results0

#results0 = run_sim(1000,15,1700,10,19.0,0.02,'power-law',1.5,True, tobs_long,'sim_results_Jun07/SF_quest_samp_power_law_gamma15.txt')
#del results0

################################################################################
#running DRW simulations with regular sampling

results0 = run_sim(1000,15,1700,10,19.0,0.02,'DRW',50,False,0.0,'sim_results_Jun07/SF_reg_samp_dl1700_dt_10_DRW_tau50.txt')
del results0

results0 = run_sim(1000,15,1700,10,19.0,0.02,'DRW',200,False,0.0,'sim_results_Jun07/SF_reg_samp_dl1700_dt_10_DRW_tau200.txt')
del results0

results0 = run_sim(1000,15,1700,10,19.0,0.02,'DRW',100,False,0.0,'sim_results_Jun07/SF_reg_samp_dl1700_dt_10_DRW_tau100.txt')
del results0

results0 = run_sim(1000,15,1700,10,19.0,0.02,'DRW',300,False,0.0,'sim_results_Jun07/SF_reg_samp_dl1700_dt_10_DRW_tau300.txt')
del results0

results0 = run_sim(1000,15,1700,10,19.0,0.02,'DRW',400,False,0.0,'sim_results_Jun07/SF_reg_samp_dl1700_dt_10_DRW_tau400.txt')
del results0


################################################################################
#running DRW simulations with QUEST sampling

results0 = run_sim(1000,15,1700,10,19.0,0.02,'DRW',50,True, tobs_long,'sim_results_Jun07/SF_quest_samp_DRW_tau50.txt')
del results0

results0 = run_sim(1000,15,1700,10,19.0,0.02,'DRW',200,True, tobs_long,'sim_results_Jun07/SF_quest_samp_DRW_tau200.txt')
del results0

results0 = run_sim(1000,15,1700,10,19.0,0.02,'DRW',100,True, tobs_long,'sim_results_Jun07/SF_quest_samp_DRW_tau100.txt')
del results0

results0 = run_sim(1000,15,1700,10,19.0,0.02,'DRW',300,True, tobs_long,'sim_results_Jun07/SF_quest_samp_DRW_tau300.txt')
del results0

results0 = run_sim(1000,15,1700,10,19.0,0.02,'DRW',400,True, tobs_long,'sim_results_Jun07/SF_quest_samp_DRW_tau400.txt')
del results0
'''
