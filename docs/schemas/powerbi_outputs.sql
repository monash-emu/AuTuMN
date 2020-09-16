CREATE TABLE powerbi_outputs (
	-- Calibration chain ID
	chain BIGINT, 
	-- Calibration iteration ID
	run BIGINT, 
	-- Model scenario
	scenario BIGINT, 
	-- Model times
	times FLOAT, 
	-- Compartment value
	value FLOAT, 
	-- Compartment age group
	agegroup TEXT,
	-- Compartment clinical strata
	clinical TEXT, 
	-- Compartment name
	compartment TEXT
);