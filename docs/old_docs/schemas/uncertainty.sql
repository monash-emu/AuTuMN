CREATE TABLE uncertainty (
	-- Uncertainy quantile
	quantile FLOAT, 
	-- Model scenario
	scenario BIGINT,
	-- Time 
	-- TODO: Change column name to 'times' 
	time FLOAT, 
	-- Derived output name
	type TEXT, 
	-- Value of derived ouput quantile
	value FLOAT
);