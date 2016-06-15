import unittest
import numpy as np
import numpy.testing as npt
from roadprofile import _create_dropouts_index, _iter_dropout_intervals, interpolate_dropouts, iter_intervals, _calculate_msd, calculate_mpd

class TestCreateDropoutsIndex(unittest.TestCase):
    dropout_criteria = 999

    def insert_invalids(self, x, y, invalid_intervals):
        for start, end in invalid_intervals:
            y[start:end] = self.dropout_criteria

    def execute_testprocedure(self, invalid_intervals):
        x = np.linspace(1, 10, 10)
        y = np.zeros(x.shape)
        self.insert_invalids(x, y, invalid_intervals)
        drop_outs = _create_dropouts_index(y, self.dropout_criteria)
        for start, end in _iter_dropout_intervals(drop_outs):
            self.assertTrue(all(y[start:end] == self.dropout_criteria))

    def test_single_point(self):
        self.execute_testprocedure({(2,3)})

    def test_two_single_points(self):
        self.execute_testprocedure({
            (2,2), (4,4)})

    def test_two_consecutive_points(self):
        self.execute_testprocedure({
            (2,3)})

    def test_three_consecutive_points(self):
        self.execute_testprocedure({
            (2,4)})

    def test_two_consecutive_beginning(self):
            self.execute_testprocedure({
                (0,2)})

    def test_single_point_beginning(self):
            self.execute_testprocedure({
                (0,1)})

    def test_two_consecutive_end(self):
            self.execute_testprocedure({
                (8,10)})

    def test_single_point_end(self):
            self.execute_testprocedure({
                (9,10)})

class TestCreateDropoutsIndexHavingNaN(TestCreateDropoutsIndex):
    dropout_criteria = float('Nan')

    def execute_testprocedure(self, invalid_intervals):
        x = np.linspace(1, 10, 10)
        y = np.zeros(x.shape)
        self.insert_invalids(x, y, invalid_intervals)
        drop_outs = _create_dropouts_index(y, self.dropout_criteria)
        for start, end in _iter_dropout_intervals(drop_outs):
            self.assertTrue(all(np.isnan(y[start:end])))

class TestInterpolateDropouts(TestCreateDropoutsIndex):
    def execute_testprocedure(self, invalid_intervals):
        x = np.arange(1, 20)
        y_org = x * 1337 + 1.337
        y = y_org.copy()
        self.insert_invalids(x, y, invalid_intervals)
        interpolate_dropouts(x, y, self.dropout_criteria)
        npt.assert_almost_equal(y, y_org)


class TestAlwaysEqualOrStrictlyLargerThanLengthIntervals(unittest.TestCase):
    def _test_interval(self, data_in, test_out=None):
        data_in = np.array(data_in)
        if test_out is None: test_out = [data_in]
        data_out = [data_in[idx_s:idx_e] for idx_s, idx_e in iter_intervals(data_in, 0.1)]
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
    def test__calculate_msd_include_50mm_point_in_first_interval(self):
        x = np.array([0, 0.025, 0.05, 0.075, 0.1])
        y = np.array([0, 0, 1, 0, 2])
        self.assertEqual((1, 2), _calculate_msd(x ,y))

    def test__calculate_msd_include_51mm_point_in_first_interval(self):
        x = np.array([0, 0.025, 0.051, 0.075, 0.1])
        y = np.array([0, 0, 1, 0, 2])
        self.assertEqual((0, 2), _calculate_msd(x ,y))

    def test_calculate_mpd(self):
        x = np.array([0, 0.025, 0.075, 0.1])
        y = np.array([2, -2, -2, 2])
        mpd_out = calculate_mpd(x, y)
        npt.assert_array_almost_equal(np.array([2]), mpd_out)

if __name__ == '__main__':
    unittest.main()
