import sys
from pathlib import Path
import numpy as np
from scipy.stats import poisson

# Add the hyMCMCpy directory to sys.path to allow imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Import the Params class
from hymcmcpy.params import Params
from hymcmcpy.forward_models import PoissonForwardModel
from hymcmcpy.likelihood_models import PoissonLogLikelihood

# Initialize Params with example data
params = Params(
    names=['lambda'],
    vals=[0.5],
    discrete=[False]
)


# Initialize Forward model with example params
forward_model_poisson = PoissonForwardModel(params)
print("forward model:", forward_model_poisson.compute(k=10))


# Initialize PoissonLogLikelhihood model



# Generate data
num_samples = 10000
lambda_param = 2.5
poisson_samples = np.random.poisson(lambda_param, num_samples)

# Print the parameters
print("Initialized Parameters:")
print(params)

print("\nParameter Names:", params['name'])
print("Parameter Values:", params['val'])
print("Discrete Flags:", params['discrete'])
print("first param:", params[0])
