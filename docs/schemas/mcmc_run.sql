CREATE TABLE mcmc_run (
	-- Calibration chain ID
	chain BIGINT, 
	-- Calibration iteration ID
	run BIGINT, 
	-- Log-likelihood of iteration
	loglikelihood FLOAT, 
	-- A posteriori log-likelihood of iteration
	ap_loglikelihood FLOAT, 
	-- Whether the run was accepted or not
	accept BIGINT, 
	-- Run weight, calculated from acceptance
	weight BIGINT
);
