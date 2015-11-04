
AuTuMN
======

A population dynamics model of TB transmission.

This is a set of Python models that modularizes the development of population transmission models and allows a pluggable API to other system. In particular, this is designed to interface with the Optima suite of software.

We as a team have made very good progress with the project. I have now put all the modules in the "autumn" folder. It includes the following modules - please sync 

model.py
Underlying transmission dynamic model. Running this won't show you anything in the console window 

plotting.py
Plot results 

input.xlsx
Input spreadsheet. At this stage, it only contains input data for the simple model we constructed in model.py. More data will needed to be added in parallel with adding more compartments and complexity to model.py. I am happy to update the spreadsheet as we progress with the model 

read_spreadsheet.py 
This reads data in input.xlsx and import into model.py. Again it will needed to reflect the changes we will make to model.py and consequently input.xlsx; and I am happy to do this as well. 

execution.py 
This links all the above modules together and give model outputs. We will need to run this file to see the results. 

I think the next big steps will be implementing cost-functions/macroeconomics and parameter estimations. I will focus on the former while would also like to be involved in the latter. 