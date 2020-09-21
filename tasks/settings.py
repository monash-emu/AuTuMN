import os
from autumn.constants import OUTPUT_DATA_PATH
from apps import covid_19, tuberculosis

S3_BUCKET = "autumn-data"
AWS_PROFILE = "autumn"
AWS_REGION = "ap-southeast-2"
BASE_DIR = os.path.join(OUTPUT_DATA_PATH, "remote")


APP_MAP = {
    "covid_19": covid_19,
    "tuberculosis": tuberculosis,
}
