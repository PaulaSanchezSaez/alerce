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

from lc_features.features import FeaturesComputerV2

def run_paps(df_alerts):
    dt1=150
    dt2=7

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
                ratio_1,low_1,high_1,non_zero_1, PN_flag_1 = paps.statistics(lc_1.magpsf_corr.values,lc_1.sigmapsf_corr.values, lc_1.mjd.values, dt1,dt2)
            else:
                ratio_1,low_1,high_1,non_zero_1, PN_flag_1 = -99,-99,-99,-99,-99
            #rband
            lc_2 = object_alerts.loc[object_alerts.fid==2]
            if len(lc_2.magpsf_corr.values)>5:
                ratio_2,low_2,high_2,non_zero_2, PN_flag_2 = paps.statistics(lc_2.magpsf_corr.values,lc_2.sigmapsf_corr.values, lc_2.mjd.values, dt1,dt2)
            else:
                ratio_2,low_2,high_2,non_zero_2, PN_flag_2 = -99,-99,-99,-99,-99

            feats = [oid,ratio_1,low_1,high_1,non_zero_1, PN_flag_1,ratio_2,low_2,high_2,non_zero_2, PN_flag_2]
            results.append(feats)

        except:

                logging.exception('Exception computing features with oid={}'.format(oid) )
                continue

    results = np.array(results)

    df_results = pd.DataFrame(results, columns = ['oid','paps_ratio_1','paps_low_1','paps_high_1','paps_non_zero_1', 'paps_PN_flag_1','paps_ratio_2','paps_low_2','paps_high_2','paps_non_zero_2', 'paps_PN_flag_2'])

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


        features = run_paps(alerts_df)

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
