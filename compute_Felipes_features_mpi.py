import sys
import os
import time
import traceback

from mpi4py import MPI
from joblib import Parallel, delayed
from multiprocessing import cpu_count

import sys
import numpy as np
import pandas as pd

import paps

import random
import scipy
from scipy.optimize import minimize_scalar
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
from skmisc.loess import loess
import statsmodels.stats.diagnostic as smsdia

from lc_features.features import FeaturesComputerV2


def IAR_phi_loglik(x,y,sT,delta,include_mean=False,standarized=True):
    n=len(y)
    sigma=1
    mu=0
    if standarized == False:
        sigma=np.var(y,ddof=1)
    if include_mean == True:
        mu=np.mean(y)
    d=np.diff(sT)
    delta=delta[1:n]
    phi=x**d
    yhat=mu+phi*(y[0:(n-1)]-mu)
    y2=np.vstack((y[1:n],yhat))
    cte=0.5*n*np.log(2*np.pi)
    s1=cte+0.5*np.sum(np.log(sigma*(1-phi**2)+delta**2)+(y2[0,]-y2[1,])**2/(sigma*(1-phi**2)+delta**2))
    return s1

def IAR_loglik(y,sT,delta,include_mean=False,standarized=True):
    out=minimize_scalar(IAR_phi_loglik,args=(y,sT,delta,include_mean,standarized),bounds=(0,1),method="bounded",tol=0.0001220703)
    return out.x


def comp_felipes_features(sTg,yg,yerrg):

    testarch_g = smsdia.het_arch(yg, maxlag=1)
    y2=(yg-np.mean(yg))/np.sqrt(np.var(yg,ddof=1))
    yerr2=(yerrg)/np.sqrt(np.var(yg,ddof=1))
    iar=IAR_loglik(y2**2,sTg,delta=yerr2)
    features=[testarch_g[0],testarch_g[1],iar]
    return(features)

def run_feats(df_alerts):

    oids = df_alerts.index.unique()
    results = []
    for oid in oids:

        try:
            object_alerts = df_alerts.loc[[oid]]
            enough_alerts = (
                    len(object_alerts) > 0
                    and (
                        (len(object_alerts[object_alerts.fid == 1]) > 5)
                    or (len(object_alerts[object_alerts.fid == 2]) > 5)
                    )
                )

            if not enough_alerts:
                    continue

            #gband
            lc_1 = object_alerts.loc[object_alerts.fid==1]
            if len(lc_1.magpsf_corr.values)>5:
                feats_1 = comp_felipes_features(lc_1.mjd.values,lc_1.magpsf_corr.values,lc_1.sigmapsf_corr.values)
            else:
                feats_1 = [-999,-999,-999]
            #rband
            lc_2 = object_alerts.loc[object_alerts.fid==2]
            if len(lc_2.magpsf_corr.values)>5:
                feats_2 = comp_felipes_features(lc_2.mjd.values,lc_2.magpsf_corr.values,lc_2.sigmapsf_corr.values)
            else:
                feats_2 = [-999,-999,-999]

            feats = [oid]+feats_1+feats_2
            results.append(feats)

        except:

                logging.exception('Exception computing features with oid={}'.format(oid) )
                continue

    results = np.array(results)

    df_results = pd.DataFrame(results, columns = ['oid','Arch-Test_1', 'p-value_Arch-Test_1','square_IAR_phi_1','Arch-Test_2', 'p-value_Arch-Test_2','square_IAR_phi_2'])

    return(df_results)


def process_lc(process_id, alerts_directory, features_directory):

    try:
        alerts_path = os.path.join(alerts_directory,'detections_%s.pickle' %(rank) )
        alerts_df = pd.read_pickle(alerts_path)
        alerts_df.sigmapsf_corr = alerts_df.sigmapsf_corr.astype("float64")
        alerts_df.magpsf_corr = alerts_df.magpsf_corr.astype("float64")
        alerts_df.sigmapsf_corr = alerts_df.sigmapsf_corr.astype("float64")
        alerts_df.mjd = alerts_df.mjd.astype("float64")
        alerts_df = alerts_df[(alerts_df.sigmapsf_corr > 0) & (alerts_df.sigmapsf_corr < 1.0)]
        alerts_df = alerts_df[alerts_df.rb >= 0.55]


        alerts_df.set_index('oid',inplace=True)


        features = run_feats(alerts_df)

        features.to_pickle(
            os.path.join(features_directory, 'features_%d.pkl' % process_id))
    except:
        print("Error dataframe = %d" % (process_id) , file=sys.stderr )
        traceback.print_exc()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

start = time.time()

alerts_directory = sys.argv[1]
features_directory = sys.argv[2]

process_lc(rank,alerts_directory,features_directory )

end = time.time()
total = end-start
print("process %s=%s" % ( rank, total ) )

comm.Barrier()
