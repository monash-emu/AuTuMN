# library(socialmixr)

setwd("C:/Documents and Settings/rrag0004/Models/AuTuMN_new/autumn/projects/sm_covid/common_school/scripts/mixing_data/")

get_non_comix_survey_list <- function(from_csv=TRUE){
  if (from_csv){
    included_list = read.csv("socialmixr_list.csv")  
  }else{
    survey_list = list_surveys()
    
    included_list = data.frame(row.names = c("URL", "title"))
    for (i in 1:nrow(survey_list)){
      title = survey_list$title[i]
      if (!grepl("CoMix", title, fixed=TRUE)){
        data = list(
          "URL"=survey_list$url[i],
          "title"=survey_list$title[i]
        )
        included_list = rbind(included_list, data)
      }    
    }
  }
  
  return(included_list)
}

check_school_contacts_present <- function(survey_url){
  survey = get_survey(survey_url)
  are_there_non_school = FALSE %in% unique(survey$contacts$cnt_school)
  are_there_school = TRUE %in% unique(survey$contacts$cnt_school)
  
  if (are_there_non_school && are_there_school){
    include = TRUE
  }else{
    include = FALSE
  }
  
  return(include)
  
}

check_all_surveys <- function(survey_list){
  for (i in 1:nrow(survey_list)){
    if (i>=12){
      url = survey_list$URL[i]
      title = survey_list$title[i]
      print(title)
      include = check_school_contacts_present(url)
      print(include)
      print("")
      print("")
    }
  }
}

polymod_countries = list(
  "Italy"= "ITA",
  "Germany"= "DEU",
  "Luxembourg"= "LUX",
  "Netherlands"="NLD",
  "Poland"="POL",
  "United Kingdom"="GBR",
  "Finland"="FIN",
  "Belgium"="BEL"
)
polymod_iso3s = unlist(polymod_countries[names(polymod_countries)],use.names=F)

survey_list = get_non_comix_survey_list()
iso3_list = c(survey_list$iso3, polymod_iso3s)
iso3_list = iso3_list[iso3_list!="MULTI"]

for (iso3 in iso3_list){
  
  matrices = list()
  filters = list(
    "home"=list("cnt_home"=TRUE), 
    "school"=list("cnt_school"=TRUE, "cnt_home"=FALSE), 
    "work"=list("cnt_work"=TRUE, "cnt_home"=FALSE, "cnt_school"=FALSE), 
    "other_locations"=list("cnt_home"=FALSE, "cnt_school"=FALSE, "cnt_work"=FALSE)
  )
  base_age_lower_bounds = c(0, 15, 25, 50, 70)
  
  if (iso3 %in% polymod_iso3s){  # POLYMOD country
    country_name = names(polymod_countries)[which( polymod_iso3s == iso3)]
    
  }else{  # non-POLYMOD country
    table_row = survey_list[survey_list$iso3 == iso3,]
    url = table_row$URL
    survey = get_survey(url)
    
    # potentially need to modify the age bounds based on data availability
    age_lower_bounds = base_age_lower_bounds
    n_older = 0
    while (n_older < 5){
      last_age = age_lower_bounds[length(base_age_lower_bounds)]
      n_older = length(survey$participants$part_age[survey$participants$part_age >= last_age])
      if (n_older < 5){
        age_lower_bounds = age_lower_bounds[-length(age_lower_bounds)]
      }
    }
    
  }

  for (key in names(filters)){
    key_filter = filters[[key]]
    if (iso3 == "RUS"){
      key_filter['period'] = 'regular'
    }
    
    if (iso3 %in% polymod_iso3s){
      m = contact_matrix(polymod, filter=key_filter, countries=country_name, age.limits=age_lower_bounds)
    }else{
      m = contact_matrix(survey, filter=key_filter, age.limits = age_lower_bounds)
    }
    matrices[[key]] = m$matrix
    
    n_NaN = sum(is.nan(m$matrix))
    if (n_NaN > 0){
      print("NaN found in this matrix")
      print(key_filter)
    }
  }
  matrices$all = matrices$home + matrices$school + matrices$work + matrices$other_locations
  
  dir_path = paste0("../../../../../../data/inputs/social-mixing/socialmixr_school_outputs/", iso3)
  dir.create(path=dir_path)
  
  for (key in names(matrices)){
    locname = key
    if (key == "all"){
      locname = "all_locations"
    }
    filename = paste0(dir_path, "/", locname, ".csv")
    write.csv(matrices[[key]], filename)
  }
}
