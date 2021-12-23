'''Author Anurag Kumar | anuragkumarak95@gmail.com | git/anuragkumarak95

Simple example of Fractal generation using recursive function.

What is Sierpinski Triangle?
>>The Sierpinski triangle (also with the original orthography Sierpinski), also called the Sierpinski gasket or the Sierpinski Sieve, 
is a fractal and attractive fixed set with the overall shape of an equilateral triangle, subdivided recursively into smaller 
equilateral triangles. Originally constructed as a curve, this is one of the basic examples of self-similar sets, i.e., 
it is a mathematically generated pattern that can be reproducible at any magnification or reduction. It is named after 
the Polish mathematician Wacław Sierpinski, but appeared as a decorative pattern many centuries prior to the work of Sierpinski.

Requirements(pip):
  - turtle

Python:
  - 2.6

Usage:
  - $python sierpinski_triangle.py <int:depth_for_fractal>

Credits: This code was written by editing the code from http://www.lpb-riannetrujillo.com/blog/python-fractal/

'''
import sys
import turtle

PROGNAME = 'Sierpinski Triangle'
if len(sys.argv) != 2:
    raise Exception('right format for using this script: $python fractals.py <int:depth_for_fractal>')

myPen = turtle.Turtle()
myPen.ht()
myPen.speed(5)
myPen.pencolor('red')

points = [[-175, -125], [0, 175], [175, -125]]  # size of triangle


def getMid(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)  # find midpoint


def triangle(points, depth):
    """
    This function draws a Sierpinski triangle.

    :param points: A list of 3 lists, each containing 2 coordinates for the three corners of the triangle.
    :type points: list(list(int))
        :param depth: The number of times to divide each side into smaller triangles.  If this is 0 or less, nothing will be
    drawn and the function will return None.  If it's 1 or more, it'll draw a single triangle with sides that are half as long as `points`.  If it's 2 or
    more, then those triangles will be divided into even smaller ones recursively until `depth` is 0 where they'll all finally be drawn on top of one
    another forming one big equilateral triangle with sides that are twice as long as `points`.  
        :type depth: int

        .. note :: This function uses
    some functions from my
    [Sierpinski_Triangle](http://nbviewer.ipython.org/github/ehmatthes/intro_programming/blob/master/notebooks/_solved/_solved_chaos_game2d_.ipynb)
    notebook in order to do its work so you should look at that notebook
    """
    myPen.up()
    myPen.goto(points[0][0], points[0][1])
    myPen.down()
    myPen.goto(points[1][0], points[1][1])
    myPen.goto(points[2][0], points[2][1])
    myPen.goto(points[0][0], points[0][1])

    if depth > 0:
        triangle([points[0],
                  getMid(points[0], points[1]),
                  getMid(points[0], points[2])],
                 depth - 1)
        triangle([points[1],
                  getMid(points[0], points[1]),
                  getMid(points[1], points[2])],
                 depth - 1)
        triangle([points[2],
                  getMid(points[2], points[1]),
                  getMid(points[0], points[2])],
                 depth - 1)


triangle(points, int(sys.argv[1]))
