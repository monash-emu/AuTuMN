/* List all model parameters in 'name' column*/
CREATE TABLE mcmc_params (
	-- Calibration chain ID
	chain BIGINT, 
	-- Calibration iteration ID
	run BIGINT,
	-- Parameter name 
	name TEXT, 
	-- Parameter value
	value FLOAT
);
