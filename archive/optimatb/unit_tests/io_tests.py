import os
import sys
sys.path.append("..")
import optimatb
import unittest
import numpy

import os
prepend_path = os.path.dirname(os.path.realpath(__file__)) + '/'

class TestDataIO(unittest.TestCase):

	def test_basic(self):
		x = 1
		optimatb.save(x,prepend_path+'cache')
		y = optimatb.load(prepend_path+'cache')
		self.assertTrue(optimatb.dict_equal(x,y,verbose=True))

	def test_list(self):
		x = [2]
		optimatb.save(x,prepend_path+'cache')
		y = optimatb.load(prepend_path+'cache')
		self.assertTrue(optimatb.dict_equal(x,y,verbose=True))

	def test_tuple(self):
		x = (3)
		optimatb.save(x,prepend_path+'cache')
		y = optimatb.load(prepend_path+'cache')
		self.assertTrue(optimatb.dict_equal(x,y,verbose=True))

	def test_mixed(self):
		x = [(3),[4]]
		optimatb.save(x,prepend_path+'cache')
		y = optimatb.load(prepend_path+'cache')
		self.assertTrue(optimatb.dict_equal(x,y,verbose=True))

	def test_numpy_float(self):
		x = numpy.float64(1)
		optimatb.save(x,prepend_path+'cache')
		y = optimatb.load(prepend_path+'cache')
		self.assertTrue(optimatb.dict_equal(x,y,verbose=True))

	def test_numpy_array(self):
		x = numpy.array([1,2,3])
		optimatb.save(x,prepend_path+'cache')
		y = optimatb.load(prepend_path+'cache')
		self.assertTrue(optimatb.dict_equal(x,y,verbose=True))

	def test_numpy_nan(self):
		x = numpy.nan
		optimatb.save(x,prepend_path+'cache')
		y = optimatb.load(prepend_path+'cache')
		self.assertTrue(optimatb.dict_equal(x,y,verbose=True))

	def test_numpy_array_nan(self):
		x = numpy.array([1,numpy.nan,2])
		optimatb.save(x,prepend_path+'cache')
		y = optimatb.load(prepend_path+'cache')
		self.assertTrue(optimatb.dict_equal(x,y,verbose=True))

	def test_none(self):
		x = None
		optimatb.save(x,prepend_path+'cache')
		y = optimatb.load(prepend_path+'cache')
		self.assertTrue(optimatb.dict_equal(x,y,verbose=True))

if __name__ == '__main__':
    unittest.main()