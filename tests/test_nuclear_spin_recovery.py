"""
Unit and regression test for the nuclear_spin_recovery package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import nuclear_spin_recovery


def test_nuclear_spin_recovery_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "nuclear_spin_recovery" in sys.modules
