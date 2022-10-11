# library(socialmixr)
# russia_survey = get_survey("https://doi.org/10.5281/zenodo.3415222")

setwd("C:/Documents and Settings/rrag0004/Models/AuTuMN_new/autumn/projects/sm_covid/common_school/scripts/mixing_data/")

open_matrices = list()
closed_matrices = list()

base_filters = list(
  "home"=list("cnt_home"=TRUE), 
  "school"=list("cnt_school"=TRUE, "cnt_home"=FALSE), 
  "work"=list("cnt_work"=TRUE, "cnt_home"=FALSE, "cnt_school"=FALSE), 
  "other_locations"=list("cnt_home"=FALSE, "cnt_school"=FALSE, "cnt_work"=FALSE)
)

age_lower_bounds = c(0, 25, 50)

for (key in names(base_filters)){
  key_filter = base_filters[[key]]
  
  open_filters = key_filter
  open_filters["period"]="regular"
  open_matrices[[key]] = contact_matrix(russia_survey, filter = open_filters, age.limits = age_lower_bounds)$matrix
  
  closed_filters = key_filter
  closed_filters["period"]="school_closure"
  closed_matrices[[key]] = contact_matrix(russia_survey, filter = closed_filters, age.limits = age_lower_bounds)$matrix
}

open_matrices$all_locations = open_matrices$home + open_matrices$school + open_matrices$work + open_matrices$other_locations
closed_matrices$all_locations = closed_matrices$home + closed_matrices$school + closed_matrices$work + closed_matrices$other_locations

dir_path = paste0("../../../../../../data/inputs/social-mixing/socialmixr_school_outputs/RUS_relative_rates")
dir.create(path=dir_path)

for (key in names(open_matrices)){
  filename = paste0(dir_path, "/", key, "_open.csv")
  # write.csv(open_matrices[[key]], filename)
  
  filename = paste0(dir_path, "/", key, "_closed.csv")
  #write.csv(closed_matrices[[key]], filename)
  
  rel_matrix = closed_matrices[[key]] / open_matrices[[key]]
  filename = paste0(dir_path, "/", key, "_relative.csv")
  #write.csv(rel_matrix, filename)
  print(key)
  print(rel_matrix)
  print("")
}
