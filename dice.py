# Script Name	: dice.py
# Author		: Craig Richards
# Created		: 05th February 2017
# Last Modified	: 
# Version		: 1.0

# Modifications	:

# Description	: This will randomly select two numbers, like throwing dice, you can change the sides of the dice if you wish

import random
class Die(object):
  def __init__(self, sides):
    self.sides = sides
  def roll(self):
    return random.randint(1, self.sides)

d = Die(6)
d1 = Die(6)
print (d.roll(), d1.roll())
