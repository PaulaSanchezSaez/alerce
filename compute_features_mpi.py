import sys
import os
import time

from mpi4py import MPI
from joblib import Parallel, delayed
from multiprocessing import cpu_count

import sys
import numpy as np
import pandas as pd

from lc_features.features import FeaturesComputerV2

def process_lc(process_id, alerts_directory, features_directory):

    try:
        alerts_path = os.path.join(alerts_directory,'detections_%s.pickle' %(rank) )
        alerts_df = pd.read_pickle(alerts_path)
        alerts_df.sigmapsf_corr = alerts_df.sigmapsf_corr.astype("float32")
        alerts_df.magpsf_corr = alerts_df.magpsf_corr.astype("float32")
        alerts_df = alerts_df[(alerts_df.sigmapsf_corr > 0) & (alerts_df.sigmapsf_corr < 1.0)]
        alerts_df = alerts_df[alerts_df.rb >= 0.55]

        alerts_df.set_index('oid',inplace=True)

        features_computer = FeaturesComputerV2(process_id)
        features = features_computer.execute(alerts_df)

        features.to_pickle(
            os.path.join(features_directory, 'features_%d.pkl' % process_id))
    except:
        print("Error dataframe = %d" % (process_id) , file=sys.stderr )

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
