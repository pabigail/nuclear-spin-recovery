import sys
from pathlib import Path
import pytest
sys.path.append(str(Path(__file__).resolve().parent.parent))
from hymcmcpy.params import Params


def test_empty_raises_error():
    with pytest.raises(ValueError, match="Input arrays must not be empty"):
        Params([],[],[])
