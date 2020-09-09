import pandas as pd

from autumn.db import Database

from .fetch import COVID_PHL_CSV_PATH

COVID_BASE_DATE = pd.datetime(2019, 12, 31)


# Make lists of testing facilities and associated region
FACILITY_MAP = { 'Batangas Medical Center GeneXpert Laboratory':'calabarzon', 
                 'Daniel O. Mercado Medical Center':'calabarzon',
                 'De La Salle Medical and Health Sciences Institute':'calabarzon',
                 'Greencity Medical Center':'calabarzon',
                 'Lucena United Doctors Hospital and Medical Center':'calabarzon',
                 'Mary Mediatrix Medical Center':'calabarzon',
                 'Ospital ng Imus':'calabarzon',
                 'Qualimed Hospital Sta. Rosa':'calabarzon',
                ' San Pablo College Medical Center':'calabarzon', 
                 'San Pablo District Hospital':'calabarzon',
                 'UPLB Covid-19 Molecular Laboratory':'calabarzon',
                 'Allegiant Regional Care Hospital':'central visayas',
                 'Cebu Doctors University Hospital, Inc.':'central visayas',
                 'Cebu TB Reference Laboratory - Molecular Facility for COVID-19 Testing':'central visayas',
                 'Chong Hua Hospital':'central visayas',
                 'Governor Celestino Gallares Memorial Medical Center':'central visayas',
                 'Prime Care Alpha Covid-19 Testing Laboratory':'central visayas',
                 'University of Cebu Medical Center':'central visayas',
                 'Vicente Sotto Memorial Medical Center (VSMMC)':'central visayas',
                 'Amang Rodriguez Memorial Center GeneXpert Laboratory':'metro manila',
                 'Asian Hospital and Medical Center':'metro manila',
                 'Chinese General Hospital':'metro manila',
                 'De Los Santos Medical Center':'metro manila',
                 'Dr. Jose N. Rodriguez Memorial Hospital and Sanitarium (TALA) GeneXpert Laboratory':'metro manila',
                 'Dr. Jose N. Rodriguez Memorial Hospital and Sanitarium (TALA) RT PCR':'metro manila',
                 'Fe del Mundo Medical center':'metro manila',
                 'Hi-Precision Diagnostics (QC)':'metro manila',
                 'Lung Center of the Philippines (LCP)':'metro manila',
                 'Lung Center of the Philippines GeneXpert Laboratory':'metro manila',
                 'Makati Medical Center (MMC)':'metro manila',
                 'Marikina Molecular Diagnostics Laboratory (MMDL)':'metro manila',
                 'National Kidney and Transplant Institute':'metro manila', 
                 'National Kidney and Transplant Institute GeneXpert Laboratory':'metro manila',
                 'Philippine Children\'s Medical Center':'metro manila', 
                 'Philippine Heart Center GeneXpert Laboratory':'metro manila',
                 'Safeguard DNA Diagnostics, Inc':'metro manila',
                 'San Miguel Foundation Testing Laboratory':'metro manila',
                 'Singapore Diagnostics':'metro manila',
                 'St. Luke\'s Medical Center - BGC (HB) GeneXpert Laboratory':'metro manila',
                 'St. Luke\'s Medical Center - BGC (SLMC-BGC)':'metro manila',
                 'St. Luke\'s Medical Center - Quezon City (SLMC-QC)':'metro manila',
                 'Sta. Ana Hospital - Closed System Molecular Laboratory (GeneXpert)':'metro manila',
                 'The Medical City (TMC)':'metro manila',
                 'Tondo Medical Center GeneXpert Laboratory':'metro manila',
                 'Tropical Disease Foundation':'metro manila',
                 'University of Perpetual Help DALTA Medical Center, Inc.':'metro manila',
                 'UP-PGH Molecular Laboratory':'metro manila',
                 'UP National Institutes of Health (UP-NIH)':'metro manila',
                 'UP Philippine Genome Center':'metro manila',
                 'Veteran Memorial Medical Center':'metro manila',
                 'Victoriano Luna - AFRIMS':'metro manila'
                 }

def create_region_aggregates(df):
    """
    creates aggregates for each regions and an additional 'undefined' for those not mapped.
    """

    df.replace({"facility_name":FACILITY_MAP}, inplace=True)
    df.loc[~df.facility_name.isin(["calabarzon","metro manila","central visayas"]),"facility_name"] = "undefined"
    df.report_date = pd.to_datetime(df["report_date"], infer_datetime_format=True)
    
    df = df.groupby(["report_date", "facility_name"]).sum().reset_index()
    df["date_index"] = (df.report_date - COVID_BASE_DATE).dt.days
    df.drop(['pct_positive_cumulative', 'pct_negative_cumulative'],1,inplace=True)

    return df


def preprocess_covid_phl(input_db: Database):

    df = pd.read_csv(COVID_PHL_CSV_PATH)
    df = create_region_aggregates(df)
    input_db.dump_df("covid_phl", df)

 
         







