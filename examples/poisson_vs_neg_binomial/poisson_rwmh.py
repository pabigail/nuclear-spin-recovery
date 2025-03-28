import sys
from pathlib import Path
from scipy.stats import poisson

# Add the hyMCMCpy directory to sys.path to allow imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Import the Params class
from hymcmcpy.params import Params
from hymcmcpy.forward_models import ForwardModel


# forward models for poisson process
class PoissonForwardModel(ForwardModel):
    """
    Forward model for the Poisson distribution.

    This model computes the probability mass function (PMF) of observing a 
    count `k` given a Poisson rate parameter `lambda`.

    Args:
        params (float): The Poisson rate parameter (`lambda`).
        k (int): The observed count, passed as a keyword argument.
        **kwargs: Additional keyword arguments (unused here but accepted for flexibility).

    Raises:
        ValueError: If 'k' is not provided as a keyword argument.
    """
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        if 'k' not in kwargs:
            raise ValueError("PoissonForwardModel requires 'k' as kwarg.")
        self.k = kwargs['k']


    def compute(self):
        lambda_val = self.params["val"][0]
        return poisson.pmf(self.k, lambda_val)


# Initialize Params with example data
params = Params(
    names=['lambda'],
    vals=[0.5],
    discrete=[False]
)


# Initialize Forward model with example params
forward_model_poisson = PoissonForwardModel(params, k=10)
print("forward model:", forward_model_poisson.compute())

# Print the parameters
print("Initialized Parameters:")
print(params)

print("\nParameter Names:", params['name'])
print("Parameter Values:", params['val'])
print("Discrete Flags:", params['discrete'])
print("first param:", params[0])
