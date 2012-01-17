import os
home=os.path.expanduser("~")
print home
if not os.path.exists(home+'/testdir'):
  os.makedirs(home+'/testdir')