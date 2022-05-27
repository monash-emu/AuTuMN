#
#  Victoria Cluster Aggregation Script
#  Call this directly from the base of the repo;
#  python scripts/dhhs_aggregation.py
#
#  Configuration is done in the constants below (VIC_CLUSTERS, AGGREGATE_OUTPUT ,COMMIT_SHA RUN_IDS)
#

import pandas as pd
import numpy as np

from pathlib import PurePosixPath, Path

from datetime import datetime
from time import time

import pickle

from autumn.core.runs.managed import ManagedRun

import s3fs

from autumn.core.utils.pandas import pdfilt
from autumn.core.project import get_project

from summer.utils import ref_times_to_dti

from autumn.models.covid_19.model import COVID_BASE_DATETIME

from autumn.core.db import uncertainty

from autumn.coreruns.utils import collate_columns_to_urun

def as_ordinal_runs(do_df):
    do_df = collate_columns_to_urun(do_df, True)
    uruns = do_df['urun'].unique()
    urun_map = dict( [(uruns[x],x) for x in range(len(uruns))] )
    new_idx = [urun_map[u] for u in do_df['urun']]
    do_df.drop('urun',axis=1,inplace=True)
    do_df['ordinal_id'] = new_idx
    dfpt = do_df.pivot_table(index=['scenario','ordinal_id','times'])
    return dfpt

# Cluster names as specified in autumn/settings/region.py
VIC_CLUSTERS = [
    "BARWON_SOUTH_WEST",
    "GIPPSLAND",
    "GRAMPIANS",
    "HUME",
    "LODDON_MALLEE",
    "NORTH_EAST_METRO",
    "SOUTH_EAST_METRO",
    "WEST_METRO",
]

# The output keys of interest
# These will be included for each cluster, as well as the state level aggregate
AGGREGATE_OUTPUTS = [
    "incidence",
    "notifications",
    "hospital_occupancy",
    "icu_occupancy",
    "accum_deaths",
    "infection_deaths",
    "hospital_admissions",
    "icu_admissions",
]

# Mapping of cluster id to run_id
RUN_IDS = {
    'barwon-south-west': 'covid_19/barwon-south-west/1635849471/6ed04a9',
    'gippsland': 'covid_19/gippsland/1635849364/6ed04a9',
    'grampians': 'covid_19/grampians/1635849367/6ed04a9',
    'hume': 'covid_19/hume/1635849378/6ed04a9',
    'north-east-metro': 'covid_19/north-east-metro/1635850532/a0a460c',
    'south-east-metro': 'covid_19/south-east-metro/1635850494/a0a460c',
    'loddon-mallee': 'covid_19/loddon-mallee/1635849425/6ed04a9',
    'west-metro': 'covid_19/west-metro/1635850489/a0a460c'
}

# Used for filename only (ie doesn't affect data) unless specified below - see comments in main script
COMMIT_SHA = "6ed04a9"


def create_csv():
    # This region name must be one of the clusters, but it doesn't matter which;
    # We need to get the targets from somewhere; they are assumed to be the same for all clusters,
    # otherwise the whole thing breaks anyway
    ref_project = get_project('covid_19', 'hume')
    targets = ref_project.plots

   

    cluster_s3_names = [c.lower().replace('_','-') for c in VIC_CLUSTERS]

    bucket = "autumn-data"
    base_remote = PurePosixPath(bucket) / 'covid_19'
    fs = s3fs.S3FileSystem()


    # Uncomment this section to use the old behaviour
    # ie all clusters have the same commit_sha, and only the latest run is used
    # RUN_IDS = {}
    #for cname in cluster_s3_names:
    #    cpath = base_remote / cname
        # Get most recent run for commit
    #    crun_path = sorted(fs.glob(str(cpath / '*' / COMMIT_SHA)))[-1]
    #    RUN_IDS[cname] = '/'.join(crun_path.split('/')[1:])


    managed_runs = dict([(cname, ManagedRun(RUN_IDS[cname])) for cname in cluster_s3_names])

    target_keys = [v['output_key'] for v in targets.values()]

    accum_df = None
    max_valid = 1e100

    for region, mr in managed_runs.items():
        print(f"Accumulating {region}")
        dfpt = as_ordinal_runs(mr.full_run.get_derived_outputs())
        dfpt = dfpt[target_keys]
        if accum_df is None:
            accum_df = dfpt
        else:
            accum_df = accum_df + dfpt
            
        max_valid = min(dfpt.index.get_level_values('ordinal_id').max(), max_valid)

    accum_df.dropna()

    print("Calculating uncertainty")
    agg_udf = uncertainty._calculate_mcmc_uncertainty(accum_df.dropna().reset_index(), targets)

    # Create map of s3 cluster names to CSV region names
    cvc_map = dict( [(cluster_s3_names[i],VIC_CLUSTERS[i]) for i in range(len(VIC_CLUSTERS))] )

    GRAND_COLLECTION = []

    collected = []
    for input_key in AGGREGATE_OUTPUTS:
        collected.append(pdfilt(agg_udf, f"type=={input_key}"))

    filt_agg = pd.concat(collected, ignore_index=True)
    filt_agg['time'] = ref_times_to_dti(COVID_BASE_DATETIME, filt_agg['time'])

    filt_agg['region'] = 'VICTORIA'

    GRAND_COLLECTION.append(filt_agg)

    mle_accum = None

    for region, mr in managed_runs.items():
        pbi = mr.powerbi.get_db()

        cudf = pbi.get_uncertainty()

        cmdf = cudf.melt(ignore_index=False)
        cmdf['time'] = cmdf.index

        collected = []
        for input_key in AGGREGATE_OUTPUTS:
            collected.append(pdfilt(cmdf, f"type=={input_key}"))
            
        final = pd.concat(collected, ignore_index=True)

        final['region'] = cvc_map[region]

        GRAND_COLLECTION.append(final)

        do_df = pbi.get_derived_outputs()

        if mle_accum is None:
            mle_accum = do_df
        else:
            mle_accum = mle_accum + do_df

        for input_key in AGGREGATE_OUTPUTS:
            cmdf = do_df[input_key].melt(ignore_index=False)
            cmdf['time'] = cmdf.index
            cmdf['type'] = input_key
            cmdf['region'] = cvc_map[region]
            cmdf.reset_index(drop=True, inplace=True)
            cmdf.dropna(inplace=True)

            GRAND_COLLECTION.append(cmdf)

    for input_key in AGGREGATE_OUTPUTS:
        vic_df = mle_accum[input_key].melt(ignore_index=False)
        vic_df['time'] = vic_df.index
        vic_df['type'] = input_key
        vic_df['region'] = 'VICTORIA'
        vic_df.reset_index(drop=True, inplace=True)
        vic_df.dropna(inplace=True)

        GRAND_COLLECTION.append(vic_df)

    final_df = pd.concat(GRAND_COLLECTION, ignore_index=True)

    final_df['time'] = final_df['time'].dt.date

    out_df = final_df[['region','scenario','time','type','value','quantile']]

    cur_time = datetime.fromtimestamp(time())
    
    dt_str = cur_time.strftime("%Y-%m-%dT%H-%M-%S")

    csv_fn = f"vic-forecast-{COMMIT_SHA}-{dt_str}.csv"

    out_df.to_csv(csv_fn, index=False)

    remote_path = PurePosixPath(bucket) / 'dhhs' / csv_fn

    # Upload to S3FS
    # Requires you have appropriate local environment variables set up for AWS access
    fs.put_file(csv_fn, remote_path)
    
    # Optional dump as pickle (if you want to examine the data locally)
    #out_df.to_pickle(f"vic-forecast-{COMMIT_SHA}-{dt_str}.pkl")

if __name__ == "__main__":
    create_csv()

