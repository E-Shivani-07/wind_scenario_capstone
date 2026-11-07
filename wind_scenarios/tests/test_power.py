import unittest

import numpy as np

from tests.helpers import small_windows  # noqa: F401  (sets sys.path)
from windgen.power import capacity_factor, hub_height_speed, wind_to_power


class TestPower(unittest.TestCase):
    def test_curve_regions(self):
        cf = capacity_factor(np.array([0.0, 3.0, 7.5, 12.0, 20.0, 25.0, 30.0]))
        self.assertEqual(cf[0], 0.0)
        self.assertEqual(cf[1], 0.0)
        self.assertTrue(0 < cf[2] < 1)
        np.testing.assert_allclose(cf[3:5], 1.0)
        np.testing.assert_allclose(cf[5:], 0.0)  # cut-out

    def test_monotone_below_rated_and_hub_height(self):
        v = np.linspace(0, 12, 50)
        self.assertTrue((np.diff(capacity_factor(v)) >= 0).all())
        self.assertAlmostEqual(float(hub_height_speed(5.0, hub=10.0)), 5.0)
        self.assertGreater(float(hub_height_speed(5.0, hub=100.0)), 5.0)
        self.assertEqual(wind_to_power(np.ones((2, 3, 4))).shape, (2, 3, 4))
        self.assertAlmostEqual(float(wind_to_power(10.0, hub=10.0, rated=10.0)), 1.0)


if __name__ == "__main__":
    unittest.main()
