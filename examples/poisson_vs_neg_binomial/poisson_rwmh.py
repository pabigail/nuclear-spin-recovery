import sys
from pathlib import Path

# Add the hyMCMCpy directory to sys.path to allow imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Import the Params class
from hymcmcpy.params import Params

# Initialize Params with example data
params = Params(
    names=['lambda'],
    vals=[0.5],
    discrete=[False]
)

# Print the parameters
print("Initialized Parameters:")
print(params)

print("\nParameter Names:", params['name'])
print("Parameter Values:", params['val'])
print("Discrete Flags:", params['discrete'])
print("first param:", params[0])
