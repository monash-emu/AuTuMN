import sim
import numpy

skip_keys = ['UUID']

def dict_equal(d1,d2,verbose=0,debug_string=''):
	# Check if two dictionaries contain the same stuff

	# Check that they're the same type
	if isinstance(d1,str):
		d1 = unicode(d1)
	if isinstance(d2,str):
		d2 = unicode(d2)

	if type(d1) != type(d2):
		if verbose:
			print '%s Types of %s and %s are different: they are %s and %s' % (debug_string,d1,d2,type(d1),type(d2))
		return False

	if isinstance(d1,sim.Sim):
		return d1 == d2

	if d1 is None and d2 is None:
		if verbose:
			print '%s %s and %s are None' % (debug_string,d1,d2)
		return True

	# If they are dictionaries, check all of their fields
	if isinstance(d1,dict):
		if d1.viewkeys() != d2.viewkeys():
			if verbose:
				print '%s Keys do not match: d1=%s,d2=%s' % (debug_string,d1.viewkeys(),d2.viewkeys())
			return False
		else:
			return all([dict_equal(d1[k],d2[k],verbose=verbose,debug_string='%s dict[%s]->' % (debug_string,k)) for k in d1.keys() if k not in skip_keys]) # Need to skip the UUID
	
	# Vector reduction for ndarrays
	elif isinstance(d1,(float,numpy.ndarray)):
		# We need to check array equality considering NaN==NaN to be true
		return isequalwithequalnans(d1,d2)

	# We might have a list of ndarrays 
	elif isinstance(d1,list):
		if len(d1) != len(d2):
			if verbose:
				print "%s List length doesn't match" % (debug_string)
			return False
		rval = all([dict_equal(d1[x],d2[x],verbose=verbose,debug_string='%s list[%s]->' % (debug_string,x)) for x in xrange(0,len(d1))]) 
		if verbose and not rval:
			print d1[1]
			print d2[1]
			print "%s List items don't match"  % (debug_string)
			return False
		return rval
	# Direct equality
	elif isinstance(d1,(tuple,int,str,unicode)):
		return d1 == d2
	else:
		raise Exception("%s Do not know how to compare objects of type %s" % (debug_string,type(d1)))

def isequalwithequalnans(a,b):
	# A close of Matlab for testing equality of arrays containing NaNs
	# Note that the type of a and b is *assumed* to be the same
	# i.e. this function will return true if the type conversion leads to equality
	# e.g. isequalwithequalnans(1,1.0) -> True
	if isinstance(a,float):
		if numpy.isnan(a) and numpy.isnan(b):
			return True
		else:
			return a==b

	elif isinstance(a,numpy.ndarray):
		if not numpy.all(a.size == b.size): # If they are different sizes, they aren't equal
			return False
		if a.size == 0:
			return True # Two empty arrays are equal

		# Now check if the non NaN entries are equal
		non_nan_equal =  numpy.ma.all(numpy.ma.masked_where(numpy.isnan(a), a) == numpy.ma.masked_where(numpy.isnan(b), b)) 
		if not non_nan_equal:
			return False

		# Finally, check that the NaNs are in the same place
		return numpy.array_equal(numpy.isnan(a),numpy.isnan(b))
	else:
		raise Exception('Unrecognized type')

