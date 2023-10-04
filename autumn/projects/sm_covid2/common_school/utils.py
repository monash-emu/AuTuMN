import yaml
import os
from autumn.settings.folders import PROJECTS_PATH

SCHOOL_ISO2_LIST = [
    'AW', 'AF', 'AO', 'AL', 'AE', 'AR', 'AM', 'AG', 'AU', 'AT', 'AZ', 'BI', 'BE', 'BJ', 'BF', 'BD', 'BG', 'BH', 'BS', 
    'BA', 'BY', 'BZ', 'BO', 'BR', 'BB', 'BN', 'BW', 'CF', 'CA', 'CH', 'CL', 'CN', 'CI', 'CM', 'CD', 'CG', 'CO', 'KM', 
    'CV', 'CR', 'CU', 'CW', 'CY', 'CZ', 'DE', 'DJ', 'DK', 'DO', 'DZ', 'EC', 'EG', 'ES', 'EE', 'ET', 'FI', 'FJ', 'FR', 
    'GA', 'GB', 'GE', 'GH', 'GN', 'GM', 'GQ', 'GR', 'GD', 'GT', 'GY', 'HN', 'HR', 'HT', 'HU', 'ID', 'IN', 'IE', 'IR', 
    'IQ', 'IS', 'IL', 'IT', 'JM', 'JO', 'JP', 'KZ', 'KE', 'KG', 'KH', 'KI', 'KR', 'KW', 'LA', 'LB', 'LR', 'LY', 'LC', 
    'LK', 'LS', 'LT', 'LU', 'LV', 'MA', 'MD', 'MV', 'MX', 'MK', 'ML', 'MT', 'MM', 'ME', 'MN', 'MZ', 'MR', 'MU', 'MW', 
    'MY', 'NA', 'NE', 'NG', 'NI', 'NL', 'NO', 'NP', 'NZ', 'OM', 'PK', 'PA', 'PE', 'PH', 'PG', 'PL', 'PT', 'PY', 'PS', 
    'QA', 'RO', 'RU', 'RW', 'SA', 'SD', 'SN', 'SG', 'SB', 'SL', 'SV', 'RS', 'SS', 'ST', 'SR', 'SK', 'SI', 'SE', 'SZ', 
    'SC', 'SY', 'TD', 'TG', 'TH', 'TL', 'TT', 'TN', 'TR', 'TZ', 'UG', 'UA', 'UY', 'US', 'UZ', 'VC', 'VE', 'VN', 'ZA', 
    'ZM', 'ZW'
]