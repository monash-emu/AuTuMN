CREATE TABLE mcmc_run (
	chain BIGINT, 
	run BIGINT, 
	loglikelihood FLOAT, 
	ap_loglikelihood FLOAT, 
	accept BIGINT, 
	weight BIGINT
);
