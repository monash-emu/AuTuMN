library(RSQLite)
setwd("C:/Users/rrag0004/Models/jtrauer_AuTuMN/autumn_from_summer/mcmc_postprocessing")
path_to_databases = '../mongolia/mcmc_chistmas_2019/'
sqlite.driver <- dbDriver("SQLite")

ylims = list('contact_rate'=c(10,20), 'adult_latency_adjustment'=c(2,6), 'dr_amplification_prop_among_nonsuccess'=c(.15,.25),
             'self_recovery_rate'=c(.18,.29), 'tb_mortality_rate'=c(.33,.44), 'rr_transmission_recovered'=c(.8,1.2), 'cdr_multiplier'=c(.66,1.5))


db_files = list.files(path_to_databases)

load_databases <- function(){
  loaded_dbs=list()
  i = 0
  for (db_file in db_files){
    i = i+1
    filename = paste(path_to_databases,db_file,sep='')
    db <- dbConnect(sqlite.driver,
                    dbname = filename)
    db_name = paste("db",i,sep='')
    loaded_dbs[[db_name]] = list()
    for (table_name in c('outputs', 'derived_outputs', 'mcmc_run')){
      loaded_dbs[[db_name]][[table_name]] = dbReadTable(db, table_name)
    }
    # make actual trace dataframe
    loaded_dbs[[db_name]]$mcmc_trace = make_actual_trace_dataframe(loaded_dbs[[db_name]]$mcmc_run)
    dbDisconnect(db)
  }
  return(loaded_dbs)
}

plot_traces_and_histograms <- function(loaded_dbs){
  param_list = colnames(loaded_dbs$db1$mcmc_trace)
  param_list = param_list[!param_list %in% c('idx', 'Scenario', 'loglikelihood', 'accept')]
  
  colours = rainbow(length(names(loaded_dbs)))
  
  # work out max number of iterations
  n_iter = 0
  for (db_index in names(loaded_dbs)){
    n_iter = max(n_iter,nrow(loaded_dbs[[db_index]]$mcmc_trace))
  }

  for (param in c(param_list, 'loglikelihood')){
    
    x11()
    if (param %in% names(ylims)){
      YLIM = ylims[[param]]
    }else if (param == 'loglikelihood'){
      YLIM=c(-1000,0)
    } else{
      YLIM=NA
    }
    plot(loaded_dbs$db1$mcmc_trace[[param]], type='l', main=param,ylab='',xlab='iter', xlim=c(0,n_iter), ylim=YLIM, col=colours[1])
    
    i=1
    for (db_index in names(loaded_dbs)){
      i=i+1
      if (db_index != 'db1'){
        iters = 1:length(loaded_dbs[[db_index]]$mcmc_trace[[param]])
        lines(iters, loaded_dbs[[db_index]]$mcmc_trace[[param]],col=colours[i])
      }
    }
  }
  
  for (param in param_list){
    x = c()
    for (db_index in names(loaded_dbs)){
      x = c(x,loaded_dbs[[db_index]]$mcmc_trace[[param]] )
    }
    x11()
    if (param %in% names(ylims)){
      hist(x, xlab='', ylab='', main=param,xlim=ylims[[param]])
    }else{
      hist(x, xlab='', ylab='', main=param)
    }
  }
  
  print(length(x))
  
}

make_actual_trace_dataframe <- function(mcmc_run_table){
  trace_table = mcmc_run_table
  prev_accepted_row = trace_table[1,]
  for (i in 2:nrow(mcmc_run_table)){
    if (trace_table$accept[i] == 0){ # replace row with previous accepted
      trace_table[i,] = prev_accepted_row
    }else{
      prev_accepted_row = trace_table[i,]
    }
  }
  return(trace_table)
}

find_best_likelihood_params <- function(loaded_dbs){
  best_db_index = 'db1'
  best_run_index = 0
  best_ll = -1.e30
  for (db_index in names(loaded_dbs)){
    best_ll_this_db = max(loaded_dbs[[db_index]]$mcmc_run$loglikelihood)
    if (best_ll_this_db>best_ll){
      best_ll = best_ll_this_db
      best_db_index = db_index
      best_run_index = which.max(loaded_dbs[[db_index]]$mcmc_run$loglikelihood)
    }
  }
  str_out = paste("Best run found in database ", best_db_index, " for run_", best_run_index-1, sep='')
  
  print(str_out)
  return(loaded_dbs[[best_db_index]]$mcmc_run[best_run_index,])
  
}


find_number_of_iterations_after_burnin <- function(loaded_dbs, burn_in_values=seq(0, 400)){
  n_iterations_after_burnin = rep(0,length(burn_in_values))
  for (db_index in names(loaded_dbs)){
    n_iterations_this_db = nrow(loaded_dbs[[db_index]]$mcmc_trace)
    for (burn_in_value in burn_in_values){
      n_iter_this_db_after_burnin = max(0, n_iterations_this_db - burn_in_value)
      n_iterations_after_burnin[burn_in_value] = n_iterations_after_burnin[burn_in_value] + n_iter_this_db_after_burnin   
    }
  }  
  x11()
  plot(burn_in_values, n_iterations_after_burnin, type='l')
  return(n_iterations_after_burnin)
}


loaded_dbs = load_databases()

find_number_of_iterations_after_burnin(loaded_dbs)

plot_traces_and_histograms(loaded_dbs)

best_params = find_best_likelihood_params(loaded_dbs)
print(best_params)