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
import glob
import os


###########################################################################################################
#to read the results files
def read_sim_result(file_name,return_tau,return_obs):

    seed,nepochs,trange,tau, SFinf,pvar,exvar,err_exvar,pvar_obs,exvar_obs,err_exvar_obs, var_gp, tau_gp, var_gp_obs, tau_gp_obs,tau_iar,phi,tau_iar_obs,phi_obs,tau_car,sigma_car,tau_car_obs,sigma_car_obs,tau_fats,sigma_fats,tau_fats_obs,sigma_fats_obs = np.genfromtxt(file_name,comments="#",unpack=True)

    if return_tau and return_obs:

        med_tau_gp = np.median(tau_gp_obs)
        low_tau_gp = np.median(tau_gp_obs) - np.percentile(tau_gp_obs, 15.865)
        up_tau_gp = np.percentile(tau_gp_obs, 84.135) - np.median(tau_gp_obs)
        out_tau_gp = np.array([med_tau_gp,low_tau_gp,up_tau_gp])

        med_tau_iar = np.median(tau_iar_obs)
        low_tau_iar = np.median(tau_iar_obs) - np.percentile(tau_iar_obs, 15.865)
        up_tau_iar = np.percentile(tau_iar_obs, 84.135) - np.median(tau_iar_obs)
        out_tau_iar = np.array([med_tau_iar,low_tau_iar,up_tau_iar])

        med_tau_car = np.median(tau_car_obs)
        low_tau_car = np.median(tau_car_obs) - np.percentile(tau_car_obs, 15.865)
        up_tau_car = np.percentile(tau_car_obs, 84.135) - np.median(tau_car_obs)
        out_tau_car = np.array([med_tau_car,low_tau_car,up_tau_car])

        med_tau_fats = np.median(tau_fats_obs)
        low_tau_fats = np.median(tau_fats_obs) - np.percentile(tau_fats_obs, 15.865)
        up_tau_fats = np.percentile(tau_fats_obs, 84.135) - np.median(tau_fats_obs)
        out_tau_fats = np.array([med_tau_fats,low_tau_fats,up_tau_fats])

        return(tau[0],out_tau_gp,out_tau_iar,out_tau_car,out_tau_fats,nepochs[0],trange[0])

    elif return_tau and (not return_obs):

        med_tau_gp = np.median(tau_gp)
        low_tau_gp = np.median(tau_gp) - np.percentile(tau_gp, 15.865)
        up_tau_gp = np.percentile(tau_gp, 84.135) - np.median(tau_gp)
        out_tau_gp = np.array([med_tau_gp,low_tau_gp,up_tau_gp])

        med_tau_iar = np.median(tau_iar)
        low_tau_iar = np.median(tau_iar) - np.percentile(tau_iar, 15.865)
        up_tau_iar = np.percentile(tau_iar, 84.135) - np.median(tau_iar)
        out_tau_iar = np.array([med_tau_iar,low_tau_iar,up_tau_iar])

        med_tau_car = np.median(tau_car)
        low_tau_car = np.median(tau_car) - np.percentile(tau_car, 15.865)
        up_tau_car = np.percentile(tau_car, 84.135) - np.median(tau_car)
        out_tau_car = np.array([med_tau_car,low_tau_car,up_tau_car])

        med_tau_fats = np.median(tau_fats)
        low_tau_fats = np.median(tau_fats) - np.percentile(tau_fats, 15.865)
        up_tau_fats = np.percentile(tau_fats, 84.135) - np.median(tau_fats)
        out_tau_fats = np.array([med_tau_fats,low_tau_fats,up_tau_fats])

        return(tau[0],out_tau_gp,out_tau_iar,out_tau_car,out_tau_fats,nepochs[0],trange[0])

    else:

        return(seed,nepochs,trange,tau, SFinf,pvar,exvar,err_exvar,pvar_obs,exvar_obs,err_exvar_obs,var_gp, tau_gp, var_gp_obs, tau_gp_obs,tau_iar,phi,tau_iar_obs,phi_obs,tau_car,sigma_car,tau_car_obs,sigma_car_obs,tau_fats,sigma_fats,tau_fats_obs,sigma_fats_obs)

###########################################################################################################

def plot_all_tau(file_name,return_obs,logscale,fig_name):
    """
    function that does the plots for all the methods
    """

    list_files=np.array(sorted(glob.glob(file_name)))
    print(list_files)

    tau = np.ones(len(list_files))*-9999
    med_tau_gp = np.ones(len(list_files))*-9999
    low_tau_gp = np.ones(len(list_files))*-9999
    up_tau_gp = np.ones(len(list_files))*-9999
    med_tau_iar = np.ones(len(list_files))*-9999
    low_tau_iar = np.ones(len(list_files))*-9999
    up_tau_iar = np.ones(len(list_files))*-9999
    med_tau_car = np.ones(len(list_files))*-9999
    low_tau_car = np.ones(len(list_files))*-9999
    up_tau_car = np.ones(len(list_files))*-9999
    med_tau_fats = np.ones(len(list_files))*-9999
    low_tau_fats = np.ones(len(list_files))*-9999
    up_tau_fats = np.ones(len(list_files))*-9999

    for i in range(0,len(list_files)):
        tauf,out_tau_gpf,out_tau_iarf,out_tau_carf,out_tau_fatsf,nepochs,trange = read_sim_result(list_files[i],True,return_obs)

        tau[i] = tauf
        med_tau_gp[i] = out_tau_gpf[0]
        low_tau_gp[i] = out_tau_gpf[1]
        up_tau_gp[i] = out_tau_gpf[2]
        med_tau_iar[i] = out_tau_iarf[0]
        low_tau_iar[i] = out_tau_iarf[1]
        up_tau_iar[i] = out_tau_iarf[2]
        med_tau_car[i] = out_tau_carf[0]
        low_tau_car[i] = out_tau_carf[1]
        up_tau_car[i] = out_tau_carf[2]
        med_tau_fats[i] = out_tau_fatsf[0]
        low_tau_fats[i] = out_tau_fatsf[1]
        up_tau_fats[i] = out_tau_fatsf[2]

    #do the plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1,1,1)

    if logscale:
        ax.set_yscale('log')
        ax.set_xscale('log')

    ax.errorbar(tau,med_tau_iar,yerr=[low_tau_iar,up_tau_iar],color='seagreen',ecolor='seagreen',linestyle='None',markersize=10,marker='o',label='IAR')
    ax.errorbar(tau,med_tau_gp,yerr=[low_tau_gp,up_tau_gp],color='firebrick',ecolor='firebrick',linestyle='None',markersize=10,marker='s',label='GP')
    ax.errorbar(tau,med_tau_fats,yerr=[low_tau_fats,up_tau_fats],color='steelblue',ecolor='steelblue',linestyle='None',markersize=10,marker='o',label='turboFATS')
    ax.plot(sorted(tau),sorted(tau),'k--')

    ax.set_xlabel(r'input $\tau$')
    ax.set_ylabel(r'output $\tau$')
    plt.legend()

    plt.savefig(fig_name)
    #plt.show()


###########################################################################################################
def plot_tau(file_name,which_tau,return_obs,logscale,fig_name):
    """
    function that does the plots for every method separated
    which_tau = 'gp', 'iar', 'car', or 'fats'
    """

    list_files=np.array(sorted(glob.glob(file_name)))
    print(list_files)

    tau = np.ones(len(list_files))*-9999
    med_tau = np.ones(len(list_files))*-9999
    low_tau = np.ones(len(list_files))*-9999
    up_tau = np.ones(len(list_files))*-9999


    for i in range(0,len(list_files)):
        tauf,out_tau_gpf,out_tau_iarf,out_tau_carf,out_tau_fatsf,nepochs,trange = read_sim_result(list_files[i],True,return_obs)

        if which_tau == 'gp':
            tau[i] = tauf
            med_tau[i] = out_tau_gpf[0]
            low_tau[i] = out_tau_gpf[1]
            up_tau[i] = out_tau_gpf[2]

        elif which_tau == 'iar':
            tau[i] = tauf
            med_tau[i] = out_tau_iarf[0]
            low_tau[i] = out_tau_iarf[1]
            up_tau[i] = out_tau_iarf[2]

        elif which_tau == 'car':
            tau[i] = tauf
            med_tau[i] = out_tau_carf[0]
            low_tau[i] = out_tau_carf[1]
            up_tau[i] = out_tau_carf[2]

        elif which_tau == 'fats':
            tau[i] = tauf
            med_tau[i] = out_tau_fatsf[0]
            low_tau[i] = out_tau_fatsf[1]
            up_tau[i] = out_tau_fatsf[2]

        else:
            print("Error: wrong value for which_tau")

    #do the plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1,1,1)

    if logscale:
        ax.set_yscale('log')
        ax.set_xscale('log')

    if which_tau == 'iar': ax.errorbar(tau,med_tau,yerr=[low_tau,up_tau],color='seagreen',ecolor='seagreen',linestyle='None',markersize=10,marker='o',label='IAR')
    elif which_tau == 'car': ax.errorbar(tau,med_tau,yerr=[low_tau,up_tau],color='firebrick',ecolor='firebrick',linestyle='None',markersize=10,marker='s',label='CAR')
    elif which_tau == 'gp': ax.errorbar(tau,med_tau,yerr=[low_tau,up_tau],color='firebrick',ecolor='firebrick',linestyle='None',markersize=10,marker='s',label='GP')
    elif which_tau == 'fats': ax.errorbar(tau,med_tau,yerr=[low_tau,up_tau],color='steelblue',ecolor='steelblue',linestyle='None',markersize=10,marker='o',label='turboFATS')
    ax.plot(sorted(tau),sorted(tau),'k--')

    ax.set_xlabel(r'input $\tau$')
    ax.set_ylabel(r'output $\tau$')
    plt.legend()

    plt.savefig(fig_name)
    #plt.show()


###########################################################################################################

def plot_tau_diff_length(file_name,return_obs,logscale,fig_name):
    """
    function that plots the value of tau as a function of the lc length
    """

    list_files=np.array(sorted(glob.glob(file_name)))
    print(list_files)

    tau = np.ones(len(list_files))*-9999
    trange = np.ones(len(list_files))*-9999
    med_tau_gp = np.ones(len(list_files))*-9999
    low_tau_gp = np.ones(len(list_files))*-9999
    up_tau_gp = np.ones(len(list_files))*-9999
    med_tau_iar = np.ones(len(list_files))*-9999
    low_tau_iar = np.ones(len(list_files))*-9999
    up_tau_iar = np.ones(len(list_files))*-9999
    med_tau_car = np.ones(len(list_files))*-9999
    low_tau_car = np.ones(len(list_files))*-9999
    up_tau_car = np.ones(len(list_files))*-9999
    med_tau_fats = np.ones(len(list_files))*-9999
    low_tau_fats = np.ones(len(list_files))*-9999
    up_tau_fats = np.ones(len(list_files))*-9999

    for i in range(0,len(list_files)):
        tauf,out_tau_gpf,out_tau_iarf,out_tau_carf,out_tau_fatsf,nepochsf,trangef = read_sim_result(list_files[i],True,return_obs)

        tau[i] = tauf
        trange[i] = trangef
        med_tau_gp[i] = out_tau_gpf[0]
        low_tau_gp[i] = out_tau_gpf[1]
        up_tau_gp[i] = out_tau_gpf[2]
        med_tau_iar[i] = out_tau_iarf[0]
        low_tau_iar[i] = out_tau_iarf[1]
        up_tau_iar[i] = out_tau_iarf[2]
        med_tau_car[i] = out_tau_carf[0]
        low_tau_car[i] = out_tau_carf[1]
        up_tau_car[i] = out_tau_carf[2]
        med_tau_fats[i] = out_tau_fatsf[0]
        low_tau_fats[i] = out_tau_fatsf[1]
        up_tau_fats[i] = out_tau_fatsf[2]

    #do the plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1,1,1)

    if logscale:
        ax.set_yscale('log')
        ax.set_xscale('log')

    ax.errorbar(trange,med_tau_iar,yerr=[low_tau_iar,up_tau_iar],color='seagreen',ecolor='seagreen',linestyle='None',markersize=10,marker='o',label='IAR')
    ax.errorbar(trange,med_tau_gp,yerr=[low_tau_gp,up_tau_gp],color='firebrick',ecolor='firebrick',linestyle='None',markersize=10,marker='s',label='GP')
    ax.errorbar(trange,med_tau_fats,yerr=[low_tau_fats,up_tau_fats],color='steelblue',ecolor='steelblue',linestyle='None',markersize=10,marker='o',label='turboFATS')
    ax.plot(sorted(trange),sorted(tau),'k--')

    ax.set_xlabel(r'lc length')
    ax.set_ylabel(r'$\tau$')
    plt.legend()

    plt.savefig(fig_name)
    #plt.show()






###########################################################################################################

plot_all_tau('sim_results_Jun07/DRW_quest_samp_tau*txt',True,True,'sim_plots_Jun07/log_DRW_quest_obs.pdf')
plot_all_tau('sim_results_Jun07/DRW_quest_samp_tau*txt',False,True,'sim_plots_Jun07/log_DRW_quest_org.pdf')
plot_all_tau('sim_results_Jun07/DRW_reg_samp_dl1700*txt',True,True,'sim_plots_Jun07/log_DRW_reg_obs.pdf')
plot_all_tau('sim_results_Jun07/DRW_reg_samp_dl1700*txt',False,True,'sim_plots_Jun07/log_DRW_reg_org.pdf')

plot_all_tau('sim_results_Jun07/DRW_quest_samp_tau*txt',True,False,'sim_plots_Jun07/DRW_quest_obs.pdf')
plot_all_tau('sim_results_Jun07/DRW_quest_samp_tau*txt',False,False,'sim_plots_Jun07/DRW_quest_org.pdf')
plot_all_tau('sim_results_Jun07/DRW_reg_samp_dl1700*txt',True,False,'sim_plots_Jun07/DRW_reg_obs.pdf')
plot_all_tau('sim_results_Jun07/DRW_reg_samp_dl1700*txt',False,False,'sim_plots_Jun07/DRW_reg_org.pdf')


plot_tau('sim_results_Jun07/DRW_quest_samp_tau*txt','iar',True,True,'sim_plots_Jun07/log_DRW_quest_obs_iar.pdf')
plot_tau('sim_results_Jun07/DRW_quest_samp_tau*txt','iar',False,True,'sim_plots_Jun07/log_DRW_quest_org_iar.pdf')
plot_tau('sim_results_Jun07/DRW_quest_samp_tau*txt','iar',True,False,'sim_plots_Jun07/DRW_quest_obs_iar.pdf')
plot_tau('sim_results_Jun07/DRW_quest_samp_tau*txt','iar',False,False,'sim_plots_Jun07/DRW_quest_org_iar.pdf')

plot_tau('sim_results_Jun07/DRW_quest_samp_tau*txt','car',True,True,'sim_plots_Jun07/log_DRW_quest_obs_car.pdf')
plot_tau('sim_results_Jun07/DRW_quest_samp_tau*txt','car',False,True,'sim_plots_Jun07/log_DRW_quest_org_car.pdf')
plot_tau('sim_results_Jun07/DRW_quest_samp_tau*txt','car',True,False,'sim_plots_Jun07/DRW_quest_obs_car.pdf')
plot_tau('sim_results_Jun07/DRW_quest*txt','car',False,False,'sim_plots_Jun07/DRW_quest_org_car.pdf')

plot_tau('sim_results_Jun07/DRW_quest_samp_tau*txt','gp',True,True,'sim_plots_Jun07/log_DRW_quest_obs_gp.pdf')
plot_tau('sim_results_Jun07/DRW_quest_samp_tau*txt','gp',False,True,'sim_plots_Jun07/log_DRW_quest_org_gp.pdf')
plot_tau('sim_results_Jun07/DRW_quest_samp_tau*txt','gp',True,False,'sim_plots_Jun07/DRW_quest_obs_gp.pdf')
plot_tau('sim_results_Jun07/DRW_quest_samp_tau*txt','gp',False,False,'sim_plots_Jun07/DRW_quest_org_gp.pdf')

plot_tau('sim_results_Jun07/DRW_quest_samp_tau*txt','fats',True,True,'sim_plots_Jun07/log_DRW_quest_obs_fats.pdf')
plot_tau('sim_results_Jun07/DRW_quest_samp_tau*txt','fats',False,True,'sim_plots_Jun07/log_DRW_quest_org_fats.pdf')
plot_tau('sim_results_Jun07/DRW_quest_samp_tau*txt','fats',True,False,'sim_plots_Jun07/DRW_quest_obs_fats.pdf')
plot_tau('sim_results_Jun07/DRW_quest_samp_tau*txt','fats',False,False,'sim_plots_Jun07/DRW_quest_org_fats.pdf')


plot_tau_diff_length('sim_results_Jun07/DRW_quest_samp_dl*_tau300_*txt',True,False,'sim_plots_Jun07/DRW_quest_tau300_difflength_obs.pdf')
plot_tau_diff_length('sim_results_Jun07/DRW_quest_samp_dl*_tau300_*txt',True,True,'sim_plots_Jun07/log_DRW_quest_tau300_difflength_obs.pdf')

plot_tau_diff_length('sim_results_Jun07/DRW_quest_samp_dl*_tau300_*txt',False,False,'sim_plots_Jun07/DRW_quest_tau300_difflength_org.pdf')
plot_tau_diff_length('sim_results_Jun07/DRW_quest_samp_dl*_tau300_*txt',False,True,'sim_plots_Jun07/log_DRW_quest_tau300_difflength_org.pdf')


plot_tau_diff_length('sim_results_Jun07/DRW_reg*_tau300_*txt',True,False,'sim_plots_Jun07/DRW_reg_tau300_difflength_obs.pdf')
plot_tau_diff_length('sim_results_Jun07/DRW_reg*_tau300_*txt',True,True,'sim_plots_Jun07/log_DRW_reg_tau300_difflength_obs.pdf')

plot_tau_diff_length('sim_results_Jun07/DRW_reg*_tau300_*txt',False,False,'sim_plots_Jun07/DRW_reg_tau300_difflength_org.pdf')
plot_tau_diff_length('sim_results_Jun07/DRW_reg*_tau300_*txt',False,True,'sim_plots_Jun07/log_DRW_reg_tau300_difflength_org.pdf')
