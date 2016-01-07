# This program runs all of the unit tests in the unit_tests folder
# It should run to completion without errors if everything is working as expected
# This program should in theory always run without errors on the develop branch
# i.e. As a minimum requirement, developers should not leave the repository in a broken state
# as defined by this program

import shutil
import os
import unittest

filelist = [ f for f in os.listdir("unit_tests") if f.startswith("cache") and not f.endswith('.py') ]
for f in filelist:
    os.remove('unit_tests/'+f)



if __name__ == '__main__':
    testsuite = unittest.TestLoader().discover('unit_tests', pattern='*.py')
    # The first run through will return errors for files that are being generated
    # These are fine to ignore
    unittest.TextTestRunner(verbosity=1).run(testsuite)

    # The second run through should have no errors and display 'OK' at the end
    unittest.TextTestRunner(verbosity=1).run(testsuite)


