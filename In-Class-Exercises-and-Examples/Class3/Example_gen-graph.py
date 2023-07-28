from numpy import *
import matplotlib.pyplot as plt

x = array([1,2,3,4,5]) # array with elements
y = x**2 + 1 # do some vectorial operations on x

print( x )
print( y )

# calculates the polynomial coeficients by least squares
pol = polyfit(x,y,2)
p1 = poly1d( pol ) # encapsulates the polynomial
p2 = poly1d(polyfit(x,y,1))

print( pol )
print( p1 )
print( p2 )

y1 = p1(x) # evaluates x
y2 = p2(x)
f = plt.figure() # creates one graph object
plt.plot(x, y, "bo", x, y1, x, y2) # creates the desired graph
plt.show()
f.savefig("./In-Class-Exercises-and-Examples/Class3/fig.pdf") # saves the graph in a PDF file