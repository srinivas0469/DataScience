import os

#Returns the current working directory.
os.getcwd()

#change working directory
os.chdir("F:\\Datascience\\Python")


#Basics
x = 10
y = 3
print("10 divided by 3 is", x/y)
print("remainder after 10 divided by 3 is", x%y)

x = 100
x > 80 and x <=95
x > 35 or x < 60

x = 100
y = 10
x += y
print(x)

#Strings

mystring = "Hello Python3.6"
print(mystring)

mystring = '''Hello Python3.6'''
print(mystring)

#How to include quotes within a string?
mystring = r'Hello"Python"'
print(mystring)

mystring = 'Hi How are you?'
mystring[0]
mystring[-1]

mystring.split(' ')[0]

#List

x = [1, 2, 3, 4, 5]
x[0]
x[1]
x[4]
x[-1]
x[-2]
x[:3]

z = ['A', 4, 5.1, 'M']


x = [1, 2, 3, 4, 5]
for i in range(len(x)):
    x[i] = x[i] + 5
print(x)

for i in range(len(x)):
   x[i] += 5
print(x)

X = [1, 2, 3]
Y = [4, 5, 6]
Z = X + Y
print(Z)

X = [1, 2, 3]
Y = [4, 5, 6]

import numpy as np

Z = np.add(X, Y)
print(Z)

X = [1, 2, 3]
Z = X * 3
print(Z)

#Modify / Replace a list item
X = [1, 2, 3]
X[2]=5
print(X)

#Add / Remove a list item

X = ['AA', 'BB', 'CC']
X.append('DD')
print(X)


X = ['AA', 'BB', 'CC']
X.remove('BB')
print(X)

#Sort list
k = [124, 225, 305, 246, 259]
k.sort()
print(k)

#Tuple

K = (1,2,3)
State = ('Delhi','Maharashtra','Karnataka')

for i in State:
    print(i)

#Tuple cannot be altered
X = (1, 2, 3)
X[2]=5
 
#Dictionary

teams = {'Dave' : 'team A',
         'Tim' : 'team B',
         'Babita' : 'team C',
         'Sam' : 'team B',
         'Ravi' : 'team C'
        }
#Find Values
teams['Sam']
#Delete an item
del teams['Ravi']

#Add an item
teams['Deep'] = 'team B'
teams

#Sets
#Sets are unordered collections of simple objects.
X = set(['A', 'B', 'C'])
'A' in X
X.add('D')
X.remove('C')
Y = X.copy()
x = X
Y & X

#Functions

def sum_fun(a, b):
    result = a + b
    return result 
z = sum_fun(10, 15)

def sum_fun(a, b=0):
    result = a + b
    return result
z = sum_fun(10)

#Conditional Statements (if else)

k = 27
if k%5 == 0:
  print('Multiple of 5')
else:
  print('Not a Multiple of 5')
  
#NUMPY

import numpy as np
a = np.array([15,25,14,78,96])
a
print(a)
a.dtype
a = np.array([15,25,14,78,96],dtype = "float")
a
a.dtype

#Creating the sequence of numbers
b = np.arange(start = 20,stop = 30)  
b

c = np.arange(20,30,2)   #30 is excluded.
c 

#Reshaping the arrays 
f = np.arange(101,113)
f.reshape(3,4)
f
#reshape() does not alter the shape of the original array. 
#Thus to modify the original array we can use resize( ) 
f.resize(3,4)
f

f.reshape(3,-1)

#Missing Data

val = np.array([15,10, np.nan, 3, 2, 5, 6, 4])
val.sum()

np.isnan(val)

#2D arrays

g = np.array([(10,20,30),(40,50,60)])
#Alternatively
g = np.array([[10,20,30],[40,50,60]])
g
g.ravel()

g.ndim
g.size
g.shape

np.zeros( (2,4) )
np.zeros([2,4],dtype=np.int16)   

#To get a matrix of all random numbers from 0 to 1 we write np.empty. 
np.empty( (2,3) )  
np.ones([3,3])

np.diag([14,15,16,17])

#To create an identity matrix we can use np.eye( ) .
np.eye(5,dtype = "int")

#Some Mathematics functions 

B = np.array([[0,-20,36],[40,50,1]])
np.sin(B)

B**2
B>25

np.absolute(B)
np.sqrt(B)
np.exp(B)

A = np.arange(1,10).reshape(3,3)
A

A.sum()
A.min()
A.max()
A.mean()
A.std()   #Standard deviation
A.var()  #Variance

#In order to obtain the index of the minimum and maximum elements we use argmin( ) and argmax( ) respectively.
A.argmin()
A.argmax()

#If we wish to find the above statistics for each row or column then we need to specify the axis:
A.sum(axis=0)  # sum of each column, it will move in downward             
A.mean(axis = 0)
A.std(axis = 0)
A.argmin(axis = 0)

A.min(axis=1) # min of each row, it will move in rightwise
A.argmax(axis = 1)
A.cumsum(axis=1)

x = np.arange(10)
x[2]
x[2:5]

x = np.arange(10)
x[ : :-1]  

a = np.array([8, 3, 7, 0, 4, 2, 5, 2])
np.where(a > 4)

#Indexing with Arrays of Indices
x = np.arange(11,35,2)                    
x
i = np.array( [0,1,5,3,7,9 ] )
x[i] 
j = np.array( [ [ 0, 1], [ 6, 2 ] ] )    
x[j]      

x = np.arange(15).reshape(3,5)
x
i = np.array( [ [0,1],[2,0] ] )    # indices for the first dim
j = np.array( [ [1,1],[2,0] ] )    # indices for the second dim
x[i,j]   

x[i,2]
x[:,j]

#Splitting the arrays

z = np.arange(1,16)
np.hsplit(z,5)   # Split a into 5 arrays
np.hsplit(z,(3,5))  
