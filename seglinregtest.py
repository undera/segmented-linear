import logging
import unittest
import numpy

import seglinreg


logging.basicConfig(format='%(asctime)s\t%(message)s', level=logging.INFO)


class SegLinRegTestCase(unittest.TestCase):
    def test_something(self):
        obj = seglinreg.SegLinReg(int(numpy.random.sample() * 5) + 2)

        normal = numpy.random.standard_normal(numpy.random.sample() * 100)

        data = []
        n = 0
        for val in normal:
            n += numpy.random.sample()
            data.append((n, val))

        breakpoints, r_2 = obj.calculate(data)


if __name__ == '__main__':
    unittest.main()
