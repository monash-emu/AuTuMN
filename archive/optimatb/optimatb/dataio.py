import cPickle

def save(obj,fname):
	with open(fname,'wb') as file_data:
		cPickle.dump(obj,file_data)

def load(fname):
	with open(fname,'rb') as file_data:
		obj = cPickle.load(file_data)
	return obj

