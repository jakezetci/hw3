#!/usr/bin/env python3

import json
import sys
from numpy.testing import assert_almost_equal

with open(sys.argv[1]) as json_file:
    data = json.load(json_file)

    assert_almost_equal(data['Gauss-Newton']['H0'], 70, 1)
    assert_almost_equal(data['Levenberg-Marquardt']['H0'], 70, 1)
    assert_almost_equal(data['Gauss-Newton']['Omega'], 0.73, 1)
    assert_almost_equal(data['Levenberg-Marquardt']['Omega'], 0.73, 1)
