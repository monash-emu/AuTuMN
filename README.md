
AuTuMN
======

A population dynamics model of TB transmission.

This is a set of Python models that modularizes the development of population transmission models and allows a pluggable API to other system. In particular, this is designed to interface with the Optima suite of software.


# Files

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



# TODO

1. Get Bosco to teach us Github - @done
1.5. Get Dr Bosco Ph to teach DrDr Jim and Dr Tan Github @done
2. Start programming in Github @done
3. Change current working code for simple model over to object-oriented / something close to the final "look" of the code @done
4. Start developing transmission model
4.1 Very simple model @done
5. Link transmission model to Excel inputs spreadsheet - Tan will lead
9. Link to minimisation algorithm - will have some sub-sections
10. Link to minimisation algorithm written by UNSW Optima
6. Develop Bayesian / machine learning code for refining model inputs
7. Run code for example country
8. Link to macroeconomic model
10. Write reports 
11. Write scientific papers 


# Phillipines test set
- 2013 - 98 million susceptible
- active 0.4% 400 in 100,000
- latent guesstimate - 30%
- vary n_tbfixed_contact 10-60
- who data: http://www.who.int/tb/country/data/download/en/
