library(conmat)
library(ggplot2)

setwd("C:/Users/rrag0004/Models/AuTuMN_new/notebooks/user/rragonnet/utils/conmat")

# User options
ISO3s = c("JPN")  # list of countries for which matrices are requested
AGE_BREAKS = c(0, 10, 30, 50, 70)  # lower age limits
SETTINGS = c("all", "home", "work", "school", "other")
PLOT_MATRICES = TRUE

# Read population data exported from "notebooks/user/rragonnet/utils/conmat/export_population.ipynb"
pop_df = read.csv("pop_size.csv")


build_models <- function(){
  # Build extrapolation models for the different settings, using POLYMOD data
  # The settings are c("all", "home", "work", "school", "other")
  polymod_survey_data <- get_polymod_population()

  models = list()
  for (setting in SETTINGS){
    print(paste0("Building model for setting: ", setting))
    polymod_contact_data <- get_polymod_contact_data(setting = setting)
    
    contact_model <- fit_single_contact_model(
      contact_data = polymod_contact_data,
      population = polymod_survey_data
    )     
    models[[setting]] = contact_model
  }

  return(models) 
}

set.seed(2022 - 09 - 06)

# build extrapolation models
models = build_models()

# Create extrapolated matrices
for (iso3 in ISO3s){
  country_pop = data.frame(age=pop_df[pop_df$iso3==iso3,]$start_age, pop=pop_df[pop_df$iso3==iso3,]$population)
  country_pop = as_conmat_population(country_pop, age, pop)
  dir.create(iso3,showWarnings = FALSE)
  for (setting in SETTINGS){
    synthetic_matrix = predict_contacts(
      model = models[[setting]],
      population = country_pop,
      age_breaks = c(AGE_BREAKS, Inf)
    )
    write.csv(synthetic_matrix, paste0(iso3,"/",setting,".csv"),row.names = FALSE)

    if (PLOT_MATRICES){
      png(paste0(iso3,"/",setting,".png"),res = 300,width = 1200, height = 1200)
      plot = synthetic_matrix %>%
        predictions_to_matrix() %>%
        autoplot()
      plot = plot + labs(title=paste0(iso3, " (", setting, ")"))
      print(plot)
      dev.off()
    }    
  }
}

