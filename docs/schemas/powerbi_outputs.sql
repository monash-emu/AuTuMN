CREATE TABLE powerbi_outputs (
	chain BIGINT, 
	run BIGINT, 
	scenario BIGINT, 
	times FLOAT, 
	value FLOAT, 
	agegroup TEXT, /* You can change this to a BIGINT to save space.*/
	clinical TEXT, 
	compartment TEXT
);