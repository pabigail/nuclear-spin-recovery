import sys
from pathlib import Path
import numpy as np
import pytest 
sys.path.append(str(Path(__file__).resolve().parent.parent))
from hymcmcpy.params import Params
from hymcmcpy.forward_models import PoissonForwardModel, NegativeBinomialForwardModel
from hycmcmcpy.likelihood_models import PoissonLogLikelihoodModel, NegativeBinomialLogLikelihoodModel
