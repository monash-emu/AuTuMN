
# All available compartments
class Compartment:
    SUSCEPTIBLE = "susceptible"    
    INFECTIOUS = "infectious"    
    RECOVERED = "recovered"

# All available flows
class FlowName:
    INFECTION = "infection"    
    RECOVERY = "recovery"
   
# Routinely implemented compartments
BASE_COMPARTMENTS = [
    Compartment.SUSCEPTIBLE,
    Compartment.INFECTIOUS,
    Compartment.RECOVERED,
]
