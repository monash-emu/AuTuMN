CREATE TABLE uncertainty (
	-- Uncertainy quantile
	quantile FLOAT, 
	-- Model scenario
	scenario BIGINT,
	-- Time 
	times FLOAT, 
	-- Derived output name
	type TEXT, 
	-- Value of derived ouput quantile
	value FLOAT
);