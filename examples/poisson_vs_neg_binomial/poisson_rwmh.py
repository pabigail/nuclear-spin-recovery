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
from hymcmcpy.mcmc_algorithms.runners import MCMCRunner, MCMCIterations

# Generate data
num_data = 10000
lambda_param = 2.5
poisson_data = np.random.poisson(lambda_param, num_data)

# Initialize Params with example data
initial_params = Params(
    names=['lambda_1', 'lambda_2'],
    vals=[2.6, 3.5], # initial value
    discrete=[False, True]
)



class PoissonMCMCRunner(MCMCRunner):
    def instantiation_step(self, algorithm, **kwargs):
        return RWMHContinuousStep(self.data, self.likelihood_model, **kwargs)



# Initialize Forward model with example params
forward_model_poisson_1 = PoissonForwardModel(initial_params, ['lambda_1'])
forward_model_poisson_2 = PoissonForwardModel(initial_params, ['lambda_2'])
# print("forward model:", forward_model_poisson.compute(k=10))


# Initialize PoissonLogLikelhihood model
poisson_log_likelihood_1 = PoissonLogLikelihood(forward_model_poisson_1,
                                              poisson_data)
poisson_log_likelihood_1 = PoissonLogLikelihood(forward_model_poisson_2,
                                              poisson_data)

# print("log likelihood:", poisson_log_likelihood.log_likelihood())

total_iterations = 25
sigma_sq = 0.5

schedule = [(RWMHContinuousStep, "lambda_1", 5),
            (RWMHContinuousStep, "lambda_2", 10)]

runner = PoissonMCMCRunner(poisson_data,
                           poisson_log_likelihood_1,
                           schedule,
                           total_iterations,
                           initial_params=initial_params,
                           sigma_sq=sigma_sq)

mcmc_results = runner.run()

print(mcmc_results)



