# Linear algebra library for Python  

This module contains some useful classes and functions for dealing with linear algebra in python 2.  

---

## Overview  

- class Vector  
    - This class represents a vector of arbitrary size and operations on it.  

    **Overview about the methods:**    
        
    - constructor(components : list) : init the vector  
    - set(components : list) : changes the vector components 
    - __str__() : toString method  
    - component(i : int): gets the i-th component (start with 0)  
    - size() : gets the size of the vector (number of components)  
    - euclidLength() : returns the euclidean length of the vector.  
    - operator + : vector addition  
    - operator - : vector subtraction  
    - operator * : scalar multiplication and dot product  
    - operator == : returns true if the vectors are equal otherwise false.    
    - copy() : copies this vector and returns it.  
    - changeComponent(pos,value) : changes the specified component.  
    - norm() : normalizes this vector and returns it.  

- function zeroVector(dimension)  
    - returns a zero vector of 'dimension'  
- function unitBasisVector(dimension,pos)  
    - returns a unit basis vector with a One at index 'pos' (indexing at 0)  
- function axpy(scalar,vector1,vector2)  
    - computes the axpy operation  
- function randomVector(N,a,b)
    - returns a random vector of size N, with random integer components between 'a' and 'b'.
- class Matrix
    - This class represents a matrix of arbitrary size and operations on it.

    **Overview about the methods:**  
    
    -  __str__() : returns a string representation  
    - operator * : implements the matrix vector multiplication  
                   implements the matrix-scalar multiplication.  
    - changeComponent(x,y,value) : changes the specified component.  
    - component(x,y) : returns the specified component.  
    - width() : returns the width of the matrix  
    - height() : returns the height of the matrix  
    - operator + : implements the matrix-addition.  
    - operator - : implements the matrix-subtraction  
    - operator == : returns true if the matrices are equal otherwise false.  
- function squareZeroMatrix(N)  
    - returns a square zero-matrix of dimension NxN  
- function randomMatrix(W,H,a,b)  
    - returns a random matrix WxH with integer components between 'a' and 'b'  
---

## Documentation  

The module is well documented. You can use the python in-built ```help(...)``` function.  
For instance: ```help(Vector)``` gives you all information about the Vector-class.  
Or ```help(unitBasisVector)``` gives you all information you need about the  
global function ```unitBasisVector(...)```. If you need information about a certain  
method you type ```help(CLASSNAME.METHODNAME)```.  

---

## Usage  

You will find the module in the **src** directory called ```lib.py```. You need to  
import this module in your project. Alternatively you can also use the file ```lib.pyc``` in python-bytecode.   

---

## Tests  

In the **src** directory you can also find the test-suite, its called ```tests.py```.  
The test-suite uses the built-in python-test-framework **unittest**.  
