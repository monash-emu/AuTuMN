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

survey_list = get_non_comix_survey_list()

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
