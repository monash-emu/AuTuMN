/* List all model parameters in 'name' column*/
CREATE TABLE mcmc_params (
	chain BIGINT, 
	run BIGINT, 
	name TEXT, 
	value FLOAT
);
