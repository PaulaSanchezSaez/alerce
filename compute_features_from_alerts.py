import sys
import os

from joblib import Parallel, delayed

import numpy as np
import pandas as pd

from ztf_pipeline.features import FeaturesComputer


def process_lc(process_id, alerts_df):
    features_computer = FeaturesComputer(process_id)
    features = features_computer.execute(alerts_df)

    features.to_pickle(
        os.path.join(features_directory, 'features_batch_%d.pkl' % process_id))

alerts_filename = sys.argv[1]
features_directory = sys.argv[2]
all_alerts = pd.read_pickle(alerts_filename)
if 'oid' in all_alerts.columns:
    all_alerts.set_index('oid', inplace=True)
lc_idxs = all_alerts.index.unique()

batch_size = 1000
N_batches = int(np.ceil(len(lc_idxs)/batch_size))
print("Mini batch size: %d, Number of mini batches: %d" %(batch_size, N_batches))

Parallel(n_jobs=12, verbose=11, backend="loky")(
    delayed(process_lc)(
        k,
        all_alerts.loc[lc_idxs[k*batch_size:(k+1)*batch_size]])
    for k in range(N_batches))
