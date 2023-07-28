from numpy import *
import matplotlib.pyplot as plt

# NOTE: "EXTRA" marked lines have the code of the extra exercises

def readFromFile(fileName):
    "Reads the float data from a given file name or file path"
    return loadtxt(fileName, skiprows=1) # skips 1st row to avoid reading the header

x_data = readFromFile("./Class3/data/reg_2_tr_X.dat")
y_data = readFromFile("./Class3/data/reg_2_tr_Y.dat")
xy = list(zip(x_data, y_data)) # creates a list of tuples (with point's data)
xy.sort()

# print( x_data )
# print( y_data )
print( xy )

x = []
y = []
# loads the ordered data on split variables to use on polyfit function later on
for i in range(len(xy)):
    x.append(xy[i][0])
    y.append(xy[i][1])

# calculates the polynomial coefficients by least squares
pol = poly1d(polyfit(x,y,1))
pol2 = poly1d(polyfit(x,y,6)) # EXTRA

print( pol )
print( pol2 )

y1 = pol(x) # evaluates x_data
y2 = pol2(x) # evaluates x_data # EXTRA
f = plt.figure() # creates one graph object
# plt.plot(x, y, "ro", x ,y1) # creates the desired graph
plt.plot(x, y, "ro", x ,y1, x, y2) # creates the desired graph with both polynomials # EXTRA
plt.show()
f.savefig("fig.pdf") # saves the graph in a PDF file