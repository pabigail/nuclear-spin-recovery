import sys
from pathlib import Path
import numpy as np
from scipy.stats import poisson

# Add the hyMCMCpy directory to sys.path to allow imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from hymcmcpy.params import Params
from hymcmcpy.forward_models import PoissonForwardModel
from hymcmcpy.likelihood_models import PoissonLogLikelihood
from hymcmcpy.mcmc_algorithms.steps import RWMHContinuousStep

# Generate data
num_data = 10000
lambda_param = 2.5
poisson_data = np.random.poisson(lambda_param, num_data)

# Initialize Params with example data
params = Params(
    names=['lambda'],
    vals=[2.6], # initial value
    discrete=[False]
)

# Initialize Forward model with example params
forward_model_poisson = PoissonForwardModel(params)
print("forward model:", forward_model_poisson.compute(k=10))


# Initialize PoissonLogLikelhihood model
poisson_log_likelihood = PoissonLogLikelihood(forward_model_poisson,
                                              poisson_data)

print("log likelihood:", poisson_log_likelihood.log_likelihood())



# Take one step of RWMH
sigma_sq = 1
rwmh_step = RWMHContinuousStep(poisson_data,
                                            poisson_log_likelihood,
                                            sigma_sq)

second_params, accept_1 = rwmh_step.one_step(params)

third_params, accept_2 = rwmh_step.one_step(second_params)

print(rwmh_step.nickname)

# Print the parameters
print("Initialized Parameters:")
print(params)

# print after step
print("Updated parameters:")
print(second_params)

# print if accepted or rejected
print("accept? 1", accept_1)

# print after step
print("Updated parameters:")
print(third_params)

print("accept? 2", accept_2)

