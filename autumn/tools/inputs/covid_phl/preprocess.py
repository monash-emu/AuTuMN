import pandas as pd

from autumn.tools.db import Database

from .fetch import COVID_PHL_CSV_PATH, COVID_PHL_VAC_PATH

from autumn.settings.constants import COVID_BASE_DATETIME

# Make lists of testing facilities and associated region
FACILITY_MAP = {
    "Batangas Medical Center GeneXpert Laboratory": "calabarzon",
    "Batangas Medical Center RT PCR Laboratory": "calabarzon",
    "Calamba Medical Center": "calabarzon",
    "Daniel O. Mercado Medical Center": "calabarzon",
    "De La Salle Medical and Health Sciences Institute": "calabarzon",
    "Fatima University Medical Center Antipolo Corp.": "calabarzon",
    "GentriMed Molecular Laboratory": "calabarzon",
    "Laguna Holy Family Hospital, Inc.": "calabarzon",
    "Lucena United Doctors Hospital and Medical Center": "calabarzon",
    "Mary Mediatrix Medical Center": "calabarzon",
    "Ospital ng Imus": "calabarzon",
    "PRC- Batangas Chapter Molecular Laboratory": "calabarzon",
    "Qualimed Hospital Sta. Rosa": "calabarzon",
    "Rakkk Prophet Medical Center GeneXpert Laboratory": "calabarzon",
    "San Pablo College Medical Center": "calabarzon",
    "San Pablo District Hospital": "calabarzon",
    "Sta. Rosa CHO GeneXpert Laboratory": "calabarzon",
    "Stark Molecular Laboratory, Inc.": "calabarzon",
    "UPLB Covid-19 Molecular Laboratory": "calabarzon",
    "Allegiant Regional Care Hospital": "central visayas",
    "BioPath Clinical Diagnostics, Inc - CEBU": "central visayas",
    "BioPath Clinical Diagnostics, Inc. - AFRIMS": "central visayas",
    "Biopath Clinical Diagnostics, Inc. - AFRIMS GeneXpert Laboratory": "central visayas",
    "Bohol Containerized PCR Laboratory": "central visayas",
    "Cebu Doctors University Hospital, Inc.": "central visayas",
    "Cebu TB Reference Laboratory - GeneXpert": "central visayas",
    "Cebu TB Reference Laboratory - Molecular Facility for COVID-19 Testing": "central visayas",
    "Chong Hua Hospital": "central visayas",
    "Governor Celestino Gallares Memorial Medical Center": "central visayas",
    "Negros Oriental Provincial Hospital": "central visayas",
    "Philippine Red Cross - Cebu Chapter": "central visayas",
    "Prime Care Alpha Covid-19 Testing Laboratory": "central visayas",
    "University of Cebu Medical Center": "central visayas",
    "Vicente Sotto Memorial Medical Center (VSMMC)": "central visayas",
    "A Star Laboratories": "metro manila",
    "AL Molecular Diagnostic Laboratory": "metro manila",
    "Amang Rodriguez Memoral Medical Center (RT PCR)": "metro manila",
    "Amang Rodriguez Memorial Center GeneXpert Laboratory": "metro manila",
    "Amosup Seamen's Hospital": "metro manila",
    "Army General Hospital Molecular Laboratory": "metro manila",
    "Asian Hospital and Medical Center": "metro manila",
    "BioPath Clinical Diagnostic, Inc.": "metro manila",
    "BioPath Clinical Diagnostics, Inc. (E. Rodriguez)": "metro manila",
    "BioPath Clinical Diagnostics, Inc. (E. Rodriguez) GeneXpert Laboratory": "metro manila",
    "BioPath Clinical Diagnostics, Inc. GeneXpert Laboratory": "metro manila",
    "Caloocan City North Medical Center": "metro manila",
    "Cardinal Santos Medical Center": "metro manila",
    "Chinese General Hospital": "metro manila",
    "De Los Santos Medical Center": "metro manila",
    "Detoxicare Molecular Diagnostics Laboratory": "metro manila",
    "Dr. Jose N. Rodriguez Memorial Hospital and Sanitarium (TALA) GeneXpert Laboratory": "metro manila",
    "Dr. Jose N. Rodriguez Memorial Hospital and Sanitarium (TALA) RT PCR": "metro manila",
    "Fe del Mundo Medical center": "metro manila",
    "First Aide Diagnostic Center": "metro manila",
    "Health Delivery Systems": "metro manila",
    "Health Metrics": "metro manila",
    "Hero Laboratories": "metro manila",
    "Hi-Precision Diagnostics (QC)": "metro manila",
    "IOM - Manila Health Centre Laboratory - GeneXpert": "metro manila",
    "Jose R. Reyes Memorial Medical Center GeneXpert Laboratory": "metro manila",
    "JT Cenica Medical Health System": "metro manila",
    "Kairos Diagnostics Laboratory": "metro manila",
    "Kaiser Medical Center Inc.": "metro manila",
    "Las Pinas General Hospital and Satellite Trauma Center": "metro manila",
    "Las Pinas General Hospital and Satellite Trauma Center GeneXpert Laboratory": "metro manila",
    "Lifecore Biointegrative Inc.": "metro manila",
    "Lung Center of the Philippines (LCP)": "metro manila",
    "Lung Center of the Philippines GeneXpert Laboratory": "metro manila",
    "Makati Medical Center (MMC)": "metro manila",
    "Manila Diagnostic Center for OFW Inc.": "metro manila",
    "Manila Doctors Hospital": "metro manila",
    "Manila Healthtek Inc.": "metro manila",
    "Marikina Molecular Diagnostics Laboratory (MMDL)": "metro manila",
    "National Kidney and Transplant Institute": "metro manila",
    "National Kidney and Transplant Institute GeneXpert Laboratory": "metro manila",
    "New World Diagnostic Premium Medical Branch": "metro manila",
    "Pasig City Molecular Laboratory": "metro manila",
    "Philippine Airport Diagnostic Laboratory": "metro manila",
    "Philippine Children's Medical Center": "metro manila",
    "Philippine General Hospital GeneXpert Laboratory": "metro manila",
    "Philippine Heart Center GeneXpert Laboratory": "metro manila",
    "Philippine Red Cross - Port Area": "metro manila",
    "Philippine Red Cross (PRC)": "metro manila",
    "Philippine Red Cross Logistics & Multipurpose Center": "metro manila",
    "PNP Crime Laboratory": "metro manila",
    "PNP General Hospital": "metro manila",
    "QR Medical Laboratories Inc. (VitaCare)": "metro manila",
    "Quezon City Molecular Diagnostics Laboratory": "metro manila",
    "Research Institute for Tropical Medicine (RITM)": "metro manila",
    "Rizal Medical Center GeneXpert Laboratory": "metro manila",
    "Safeguard DNA Diagnostics, Inc": "metro manila",
    "San Lazaro Hospital (SLH)": "metro manila",
    "San Miguel Foundation Testing Laboratory": "metro manila",
    "Singapore Diagnostics": "metro manila",
    "South Super Hi Way Molecular Diagnostic Lab": "metro manila",
    "St. Luke's Medical Center - BGC (HB) GeneXpert Laboratory": "metro manila",
    "St. Luke's Medical Center - BGC (SLMC-BGC)": "metro manila",
    "St. Luke's Medical Center - Quezon City (SLMC-QC)": "metro manila",
    "St. Luke's Medical Center - Quezon City GeneXpert Laboratory": "metro manila",
    "Sta. Ana Hospital - Closed System Molecular Laboratory (GeneXpert)": "metro manila",
    "Sta. Ana Hospital - Closed System Molecular Laboratory (RT PCR)": "metro manila",
    "Supercare Medical Services, Inc.": "metro manila",
    "Taguig City Molecular Laboratory": "metro manila",
    "The Lord's Grace Medical and Industrial Clinic": "metro manila",
    "The Medical City (TMC)": "metro manila",
    "The Premier Molecular Diagnostics": "metro manila",
    "Tondo Medical Center GeneXpert Laboratory": "metro manila",
    "Tropical Disease Foundation": "metro manila",
    "University of Perpetual Help DALTA Medical Center, Inc.": "metro manila",
    "University of the East Ramon Magsaysay Memorial Medical Center": "metro manila",
    "UP National Institutes of Health (UP-NIH)": "metro manila",
    "UP Philippine Genome Center": "metro manila",
    "UP-PGH Molecular Laboratory": "metro manila",
    "Valenzuela Hope Molecular Laboratory": "metro manila",
    "Veteran Memorial Medical Center": "metro manila",
    "Victoriano Luna - AFRIMS": "metro manila",
    "Davao Doctors Hospital GeneXpert Laboratory": "davao city",
    "Davao One World Diagnostic Center Incorporated": "davao city",
    "Davao Regional Medical Center GeneXpert Laboratory": "davao city",
    "Davao Regional Medical Center RT PCR": "davao city",
    "Southern Philippines Medical Center (SPMC)": "davao city",
    "Southern Philippines Medical Center (SPMC) - Pop-up": "davao city",
}


def create_region_aggregates(df):
    """
    creates aggregates for each region and national level testing data.
    """

    # Get data out for the three main sub-regions and mark unmatched data
    df.replace({"facility_name": FACILITY_MAP}, inplace=True)
    df.loc[
        ~df.facility_name.isin(
            ["calabarzon", "metro manila", "central visayas", "davao city"]
        ),
        "facility_name",
    ] = "unmatched"
    df.report_date = pd.to_datetime(df["report_date"], infer_datetime_format=True)

    # Get national estimates and collate
    phldf = df.copy()
    phldf["facility_name"] = "philippines"
    combined_df = df.append(phldf)

    # Have to do this after national calculation
    davao_region = combined_df[combined_df.facility_name == "davao city"]
    davao_region["facility_name"] = "davao region"
    combined_df = combined_df.append(davao_region)

    # Tidy up and return
    combined_df = combined_df[combined_df.facility_name != "unmatched"]
    combined_df = (
        combined_df.groupby(["report_date", "facility_name"]).sum().reset_index()
    )
    combined_df["date_index"] = (combined_df.report_date - COVID_BASE_DATETIME).dt.days
    combined_df.drop(
        ["pct_positive_cumulative", "pct_negative_cumulative"], 1, inplace=True
    )
    return combined_df


def preprocess_covid_phl(input_db: Database):

    df = pd.read_csv(COVID_PHL_CSV_PATH)
    df = create_region_aggregates(df)
    input_db.dump_df("covid_phl", df)
    df = pd.read_csv(COVID_PHL_VAC_PATH)
    input_db.dump_df("covid_phl_vac", df)
