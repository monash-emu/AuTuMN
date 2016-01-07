import os
import sys
sys.path.append("..")
import optimatb
import unittest
from optimatb.defaults import meta as default_meta
from optimatb.parameters import pars as default_pars

import os
prepend_path = os.path.dirname(os.path.realpath(__file__)) + '/'

def compare_cached_object(fcn,regenerate = False):
	# This function takes in a function object that generates a saveable
	# object, which must be comparable using dict_equal

	test_name = fcn.__name__
	new_obj = fcn()
	fname = prepend_path + 'cache_' + test_name + '.tb'
	if regenerate or not os.path.isfile(fname): # Test existance more nicely
		# Write to file
		optimatb.save(new_obj,fname)
		return False # Is there a magic value to return to filter the results?
	else:
		old_obj = optimatb.load(fname)
		return optimatb.dict_equal(old_obj,new_obj,verbose=True)

class Tests(unittest.TestCase):

	def test_simple_run(self):

		def simple_run():
			s = optimatb.Sim(default_meta,default_pars)
			s.run()
			s.plot(show_wait=False)
			return s

		self.assertTrue(compare_cached_object(simple_run))

	def test_simparameter(self):

		def simparameter_run():
			s = optimatb.SimParameter(default_meta,default_pars)
			s.create_override('tbdeath',1900,2000,0.2,0.5)
			s.run()
			s.plot(show_wait=False)
			return s

		self.assertTrue(compare_cached_object(simparameter_run))

	def test_plot_overlay(self):
		s = optimatb.Sim(default_meta,default_pars)
		s.run()
		s2 = optimatb.SimParameter(default_meta,default_pars)
		s2.create_override('tbdeath',1900,2000,0.2,0.5)
		s2.run()
		optimatb.plot_overlay([s,s2],show_wait=False)

if __name__ == '__main__':
    unittest.main()