import unittest
import numpy as np
import numpy.testing as npt

from roadprofile import _calc_mpd_core, calculate_mpd, _calc_tpa_core, _create_dropouts_index, _iter_intervals_of_true, iter_intervals_by_length, interpolate_dropouts

class InterpolateDropoutsBaseTests:
    dropout_criteria = 999

    def insert_invalids(self, x, y, invalid_intervals):
        for start, end in invalid_intervals:
            if end > len(y):
                raise Exception('Interval [{},{}] too long for the test-array'.format(start,end))
            y[start:end] = self.dropout_criteria


    def test_single_point(self):
        self.execute_testprocedure({(2,3)})

    def test_two_single_points_one_point_apart(self):
        self.execute_testprocedure([(2,3), (4,5)])

    def test_two_consecutive_points(self):
        self.execute_testprocedure({(2,4)})

    def test_three_consecutive_points(self):
        self.execute_testprocedure({(2,5)})

    def test_two_consec_and_one_single(self):
        self.execute_testprocedure({(2,5), (7,8)})


class TestCreateDropoutsIndex(unittest.TestCase, InterpolateDropoutsBaseTests):
    def execute_testprocedure(self, invalid_intervals):
        x = np.linspace(1, 10, 10)
        y = np.zeros(x.shape)
        self.insert_invalids(x, y, invalid_intervals)
        drop_outs = _create_dropouts_index(y, self.dropout_criteria)
        for start, end in _iter_intervals_of_true(drop_outs):
            boollist = list(y[start:end] == self.dropout_criteria)
            self.assertListEqual(boollist, [True]*len(boollist))
        self.x = x
        self.y = y

    def test_two_consecutive_beginning(self):
        self.execute_testprocedure({(0,3)})

    def test_single_point_beginning(self):
        self.execute_testprocedure({(0,1)})

    def test_two_consecutive_end(self):
        self.execute_testprocedure({(8,10)})

    def test_single_point_end(self):
        self.execute_testprocedure({(9,10)})

class TestCreateDropoutsIndexHavingNaN(TestCreateDropoutsIndex):
    dropout_criteria = float('Nan')

    def execute_testprocedure(self, invalid_intervals):
        x = np.linspace(1, 10, 10)
        y = np.zeros(x.shape)
        self.insert_invalids(x, y, invalid_intervals)
        drop_outs = _create_dropouts_index(y, self.dropout_criteria)
        for start, end in _iter_intervals_of_true(drop_outs):
            self.assertTrue(all(np.isnan(y[start:end])))

class TestInterpolateDropouts(unittest.TestCase, InterpolateDropoutsBaseTests):
    def execute_testprocedure(self, invalid_intervals):
        x = np.arange(1, 11)
        y_org = x * 1337 + 1.337
        y = y_org.copy()
        self.insert_invalids(x, y, invalid_intervals)
        y, _ = interpolate_dropouts(x, y, self.dropout_criteria)
        npt.assert_almost_equal(y, y_org)

    def wrap_interpolate_dropouts(self, x, invalid_intervals):
        y_org = x * 1337 + 1.337
        y = y_org.copy()
        self.insert_invalids(x, y, invalid_intervals)
        y, truncated = interpolate_dropouts(x, y, self.dropout_criteria)
        return y_org, y, x, truncated

    def test_truncate_two_beginning_invalids(self):
        x = np.arange(1, 11) # This gives a distance of 1m per measurement which should trigger array truncation :)
        y_org, y, x, trunc = self.wrap_interpolate_dropouts(x, {(0, 3)})
        npt.assert_array_almost_equal((3, len(y_org)), trunc)
        npt.assert_almost_equal(y, y_org[3:])

    def test_interpolate_two_beginning_invalids(self):
        x = np.arange(1, 11) * 1e-3 # Milimeters which should trigger (non-truncating) interpolation
        y_org, y, x, trunc = self.wrap_interpolate_dropouts(x, {(0, 3)})
        y_org[:3] = y_org[3]
        npt.assert_array_almost_equal((0, len(y_org)), trunc)
        npt.assert_almost_equal(y, y_org)

    def test_interpolate_two_beginning_plus_extra_invalids(self):
        # Check that proper adjustments will be made for the remaining intervals if truncated at beginning
        x = np.arange(1, 15)
        y_org, y, x, trunc = self.wrap_interpolate_dropouts(x, {(0, 3), (5,6), (8,11), (13, 14)})
        y_org[13] = y_org[12]
        npt.assert_array_almost_equal((3, len(y_org)), trunc)
        npt.assert_almost_equal(y, y_org[3:])

    def test_interpolate_first_point(self):
        x = np.arange(1, 11)
        y_org, y, x, trunc = self.wrap_interpolate_dropouts(x, {(0, 1)})
        y_org[0] = y_org[1] # Since there is only one point no truncation will occur.
        npt.assert_array_almost_equal((0, len(y_org)), trunc)
        npt.assert_almost_equal(y, y_org)

    def test_truncate_last_two_invalids(self):
        x = np.arange(1, 11)
        y_org, y, x, trunc = self.wrap_interpolate_dropouts(x, {(8, 10)})
        npt.assert_array_almost_equal((0, 8), trunc)
        npt.assert_almost_equal(y, y_org[:8])

    def test_interpolate_last_two_invalids(self):
        x = np.arange(1, 11) * 1e-3
        y_org, y, x, trunc = self.wrap_interpolate_dropouts(x, {(8, 10)})
        y_org[8:] = y_org[7]
        npt.assert_array_almost_equal((0, len(y_org)), trunc)
        npt.assert_almost_equal(y, y_org)

    def test_interpolate_last_point(self):
        x = np.arange(1, 11)
        y_org, y, x, trunc = self.wrap_interpolate_dropouts(x, {(9, 10)})
        y_org[9] = y_org[8]
        npt.assert_array_almost_equal((0, len(y_org)), trunc)
        npt.assert_almost_equal(y, y_org)


class TestAlwaysEqualOrStrictlyLargerThanLengthIntervals(unittest.TestCase):
    def _test_interval(self, data_in, test_out=None):
        data_in = np.array(data_in)
        if test_out is None: test_out = [data_in]
        data_out = [data_in[idx_s:idx_e] for idx_s, idx_e in iter_intervals_by_length(data_in, 0.1)]
        npt.assert_array_almost_equal([len(data_out)], [len(test_out)])
        for array_test, array_out in zip(test_out, data_out):
            npt.assert_array_almost_equal(array_test, array_out)

    def test_interval_sligthly_larger(self):
        self._test_interval([0, 0.09, 0.12])

    def test_interval_slightly_less_is_not_accepted(self):
        self._test_interval([0, 0.099], [])

    def test_always_length_or_strictly_larger_than_length_intervals(self):
        self._test_interval([0, 0.09, 0.1])

    def test_exact_length_will_suffice(self):
        self._test_interval([0, 0.09, 0.1, 0.12], [[0, 0.09, 0.1]])

    def test_two_intervals_no_common_datapoints(self):
        self._test_interval(
                [0, 0.025, 0.075, 0.1, 0.11, 0.15, 0.18, 0.21],
                [[0, 0.025, 0.075, 0.1], [0.11, 0.15, 0.18, 0.21]]
                )

class TestMPDAlgorithm(unittest.TestCase):
    def test__calc_mpd_core_include_50mm_point_in_first_interval(self):
        x = np.array([0, 0.025, 0.05, 0.075, 0.1])
        y = np.array([0, 0, 1, 0, 2])
        npt.assert_almost_equal(1.5, _calc_mpd_core(x ,y))

    def test__calc_mpd_core_include_51mm_point_in_first_interval(self):
        x = np.array([0, 0.025, 0.051, 0.075, 0.1])
        y = np.array([0, 0, 1, 0, 2])
        npt.assert_almost_equal(1, _calc_mpd_core(x ,y))

    def test_calculate_mpd_1msd(self):
        x = np.array([0, 0.025, 0.075, 0.1])
        y = np.array([2, -2, -2, 2])
        x_out, mpd_out = calculate_mpd(x, y, nmean=1)
        npt.assert_array_almost_equal(np.array([2]), mpd_out)
        npt.assert_array_almost_equal(np.array([0, 0.1]), x_out)

    def test_calculate_mpd_10msd(self):
        all_x = []
        x_sub = np.array([0, 0.025, 0.075, 0.1])
        x_out_expect = [x_sub[0]]
        for n in range(10):
            all_x.append(x_sub + n * 0.11)
            if n==9:
                x_out_expect.append(all_x[-1][-1])
        x = np.array(all_x).flatten()
        y = np.array([2, -2, -2, 2] * 10)
        x_out, mpd_out = calculate_mpd(x, y)
        npt.assert_array_almost_equal(np.array([2]), mpd_out)
        npt.assert_array_almost_equal(x_out_expect, x_out)

    def test_calculate_mpd_20msd(self):
        all_x = []
        x_sub = np.array([0, 0.025, 0.075, 0.1])
        x_out_expect = [x_sub[0]]
        for n in range(20):
            all_x.append(x_sub + n * 0.11)
            if n==9 or n==19:
                x_out_expect.append(all_x[-1][-1])
        x = np.array(all_x).flatten()
        y = np.array([2, -2, -2, 2] * 20)
        x_out, mpd_out = calculate_mpd(x, y)
        npt.assert_array_almost_equal(np.array([2, 2]), mpd_out)
        npt.assert_array_almost_equal(x_out_expect, x_out)

class TestTPACoreAlgorithm(unittest.TestCase):
    y = np.array([0, 1,   0, 1,   0, 1])
    x = np.array([0, 0.5, 1, 1.5, 2, 2.5])
    tpa_val = 1.25
    thresholds = [1, 0.5, 0.75, 0.25]

    def execute_test(self, x, y):
        for threshold in self.thresholds:
            npt.assert_almost_equal(_calc_tpa_core(x, y, threshold), self.tpa_val * threshold**2, decimal=2)

    def test_different_thresholds(self):
        self.execute_test(self.x, self.y)

    def test_negative_profile_values(self):
        self.execute_test(self.x, self.y - 0.5)
        self.execute_test(self.x, self.y - 1)
        self.execute_test(self.x, self.y - 2)

    def test_positive_profile_values(self):
        self.execute_test(self.x, self.y + 2)

    def test_zero_profile(self):
        y = np.zeros(self.y.shape)
        for threshold in self.thresholds:
            npt.assert_almost_equal(_calc_tpa_core(self.x, y, threshold), 0, decimal=2)

if __name__ == '__main__':
    unittest.main()
