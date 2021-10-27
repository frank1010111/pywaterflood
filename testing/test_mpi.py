import pytest
from itertools import product
import numpy as np
from pyCRM.multiwellproductivity import (
    calc_gains_homogeneous,
    translate_locations,
    calc_A_ij,
)
import pandas as pd

raw_locs = pd.DataFrame()


def test_Aij():
    "From the worked example by Kaviani and Valkó"
    m = 1 + np.arange(300)
    results = calc_A_ij(0.233351184, 0.36666667, 0.23333333, 0.36666667, 0.5, m)
    assert pytest.approx(10.6867, abs=0.01) == results
