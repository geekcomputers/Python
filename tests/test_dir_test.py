#!/usr/bin/python
import unittest

import sys,os
pwd = os.path.dirname(os.path.realpath(__file__))
print pwd
base_dir = os.path.join(pwd,'..')
print base_dir

sys.path.append(base_dir)
# Python/__init__.py has to be present for this to work
import dir_test 
class expandTest(unittest.TestCase):
    def test_re(self):
        dir_test.mkdir_python('tests');
        self.assertEqual(os.path.exists('tests'),True)

if __name__ == '__main__':
    unittest.main()
