#!/usr/bin/python
import unittest

import sys,os
pwd = os.path.dirname(os.path.realpath(__file__))
print pwd
base_dir = os.path.join(pwd,'..')
print base_dir

sys.path.append(base_dir)
# Python/__init__.py has to be present for this to work
import script_count 
class expandTest(unittest.TestCase):
    def test_re(self):
        self.assertEqual(script_count.clear_screen(),None)
        self.assertEqual(script_count.count_files(pwd,'.pl'),0)

if __name__ == '__main__':
    unittest.main()
