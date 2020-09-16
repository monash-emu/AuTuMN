CREATE TABLE uncertainty (
	quantile FLOAT, 
	scenario BIGINT, 
	time FLOAT,  /* Change column name to 'times' */ 
	type TEXT, 
	value FLOAT
);