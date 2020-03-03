import sys
import os

from joblib import Parallel, delayed
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import paps

from lc_features.features import FeaturesComputerV2

''' Usage:
>>> python compute_features_from_alerts.py training_set_v1/alerts.pkl features_v2/features_pkl
'''

def run_paps(df_alerts):
    dt1=100
    dt2=10

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


def process_lc(process_id, alerts_df):
    features = run_paps(alerts_df)

    features.to_pickle(
        os.path.join(features_directory, 'features_batch_%d.pkl' % process_id))


alerts_filename = sys.argv[1]
features_directory = sys.argv[2]



all_alerts = pd.read_pickle(alerts_filename)
all_alerts.sigmapsf_corr = all_alerts.sigmapsf_corr.astype("float64")
all_alerts.magpsf_corr = all_alerts.magpsf_corr.astype("float64")
all_alerts.sigmapsf_corr = all_alerts.sigmapsf_corr.astype("float64")
all_alerts.mjd = all_alerts.mjd.astype("float64")
all_alerts = all_alerts[(all_alerts.sigmapsf_corr > 0) & (all_alerts.sigmapsf_corr < 1.0)]
all_alerts = all_alerts[all_alerts.rb >= 0.55]

if 'oid' in all_alerts.columns:
    all_alerts.set_index('oid', inplace=True)
lc_idxs = all_alerts.index.unique()

batch_size = 100
N_batches = int(np.ceil(len(lc_idxs)/batch_size))
print("Mini batch size: %d, Number of mini batches: %d" %(batch_size, N_batches))

Parallel(n_jobs=5, verbose=11, backend="loky")(
    delayed(process_lc)(
        k,
        all_alerts.loc[lc_idxs[k*batch_size:(k+1)*batch_size]])
    for k in range(N_batches))
