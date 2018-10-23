#Data Manipulation with pandas - Examples

import numpy as np
import pandas as pd

mydata = {'productcode': ['AA', 'AA', 'AA', 'BB', 'BB', 'BB'],
        'sales': [1010, 1025.2, 1404.2, 1251.7, 1160, 1604.8],
        'cost' : [1020, 1625.2, 1204, 1003.7, 1020, 1124]}
df = pd.DataFrame(mydata)

#Import Data
mydata= pd.read_csv("file1.csv")
mydata1  = pd.read_csv("file1.csv", header = None)
mydata2  = pd.read_csv("file1.csv", header = None, names = ['ID', 'first_name', 'salary'])
mydata = pd.read_table("C:\\Users\\Deepanshu\\Desktop\\example2.txt")
mydata  = pd.read_csv("C:\\Users\\Deepanshu\\Desktop\\example2.txt", sep ="\t")
mydata = pd.read_excel("https://www.eia.gov/dnav/pet/hist_xls/RBRTEd.xls",sheetname="Data 1", skiprows=2)
#Suppose you need to import a file that is separated with white spaces.
mydata2 = pd.read_table("http://www.ssc.wisc.edu/~bhansen/econometrics/invest.dat", sep="\s+", header = None)



df.shape
df.head(3)
df.productcode
df["productcode"]
df.loc[: , "productcode"]
df.iloc[: , 1] #To select variable by column position, you can use df.iloc function
df[["productcode", "cost"]]
df.loc[ : , ["productcode", "cost"]]

#Describe
df.describe()
df.describe(include=['O'])
df.productcode.describe()

df.sales.mean()
df.sales.median()
df.sales.count()
df.sales.min()
df.sales.max()

df1 = df[(df.productcode == "AA") & (df.sales >= 1250)]
df1 = df.query('(productcode == "AA") & (sales >= 1250)')
df.sort_values(['sales'])

df.groupby(df.productcode).mean()
df["sales"].groupby(df.productcode).mean()
df0 = pd.DataFrame({'id': [1, 1, 2, 3, 1, 2, 2]})
df0.id = df0["id"].astype('category')
df0.describe()

df['productcode'].value_counts()

#Plots
df['sales'].hist()
df.boxplot(column='sales')

#Extract Unique Values
#The unique() function shows the unique levels or categories in the dataset.
income.Index.unique()

#The nunique( ) shows the number of unique values.
income.Index.nunique()

#Generate Cross Tab
pd.crosstab(income.Index,income.State)

#Creating a frequency distribution
income.Index.value_counts(ascending = True)

#To draw the samples
income.sample(n = 5)
income.sample(frac = 0.1)

#Selecting only a few of the columns

income.loc[:,["Index","State","Y2008"]]
income.loc[:,"Index":"Y2008"]  #Selecting consecutive columns
#In the above command both Index and Y2008 are included.
income.iloc[:,0:5]  #Columns from 1 to 5 are included. 6th column not included
income[["Index","State","Y2008"]]

data = pd.DataFrame({"A" : ["John","Mary","Julia","Kenny","Henry"], "B" : ["Libra","Capricorn","Aries","Scorpio","Aquarius"]})
data 

#Renaming all the variables.
data.columns = ['Names','Zodiac Signs']
#Renaming only some of the variables.
data.rename(columns = {"Names":"Cust_Name"},inplace = True)
#replace only a particular character in the list of the column names 
income.columns = income.columns.str.replace('Y' , 'Year ')
income.columns

#Setting one column in the data frame as the index

income.set_index("Index",inplace = True)
income.head()
#Note that the indices have changed and Index column is now no more a column
income.columns
income.reset_index(inplace = True)
income.head()


#Removing the columns and rows

income.drop('Index',axis = 1)

#Alternatively
income.drop("Index",axis = "columns")
income.drop(['Index','State'],axis = 1)
income.drop(0,axis = 0)
income.drop(0,axis = "index")
income.drop([0,1,2,3],axis = 0)

#Sorting the data

income.sort_values("State",ascending = False)
income.sort_values("State",ascending = False,inplace = True)
income.Y2006.sort_values() 

income.sort_values(["Index","Y2002"]) 

#Create new variables

income["difference"] = income.Y2008-income.Y2009

#Alternatively
income["difference2"] = income.eval("Y2008 - Y2009")
income.head()
data = income.assign(ratio = (income.Y2008 / income.Y2009))
data.head()
income.describe() #for numeric variables

#To find the total count, maximum occuring string and its frequency we write include = ['object']
income.describe(include = ['object'])  #Only for strings / objects              

income.Y2008.mean()
income.Y2008.median()
income.Y2008.min()
income.loc[:,["Y2002","Y2008"]].max()

#Groupby function

income.groupby("Index").Y2008.min()
income.groupby("Index")["Y2008","Y2010"].max()

#agg( ) function is used to find all the functions for a given variable.
income.groupby("Index").Y2002.agg(["count","min","max","mean"])
income.groupby("Index")["Y2002","Y2003"].agg(["count","min","max","mean"])  

#The following command finds minimum and maximum values for Y2002 and only mean for Y2003
income.groupby("Index").agg({"Y2002": ["min","max"],"Y2003" : "mean"}) 


#Filtering

income[income.Index == "A"]

#Alternatively
income.loc[income.Index == "A",:]

income.loc[income.Index == "A","State"]
income.loc[income.Index == "A",:].State
          
income.loc[(income.Index == "A") & (income.Y2002 > 1500000),:]
income.loc[(income.Index == "A") | (income.Index == "W"),:]

#Alternatively.
income.loc[income.Index.isin(["A","W"]),:]

income.query('Y2002>1700000 & Y2003 > 1500000')

#Dealing with missing values

import numpy as np
mydata = {'Crop': ['Rice', 'Wheat', 'Barley', 'Maize'],
        'Yield': [1010, 1025.2, 1404.2, 1251.7],
        'cost' : [102, np.nan, 20, 68]}
crops = pd.DataFrame(mydata)
crops

#isnull( ) returns True and notnull( ) returns False if the value is NaN.
crops.isnull()  #same as is.na in R
crops.notnull()  #opposite of previous command.
crops.isnull().sum()  #No. of missing values. 

crops[crops.cost.isnull()] #shows the rows with NAs.
crops[crops.cost.isnull()].Crop #shows the rows with NAs in crops.Crop
crops[crops.cost.notnull()].Crop #shows the rows without NAs in crops.Crop

#To drop all the rows which have missing values in any rows we 
#use dropna(how = "any") . By default inplace = False . 
#If how = "all" means drop a row if all the elements in that row are missing

crops.dropna(how = "any").shape
crops.dropna(how = "all").shape  
crops.dropna(subset = ['Yield',"cost"],how = 'any').shape
crops.dropna(subset = ['Yield',"cost"],how = 'all').shape
crops['cost'].fillna(value = "UNKNOWN",inplace = True)
crops

#Dealing with duplicates

data = pd.DataFrame({"Items" : ["TV","Washing Machine","Mobile","TV","TV","Washing Machine"], "Price" : [10000,50000,20000,10000,10000,40000]})
data
data.loc[data.duplicated(),:]
data.loc[data.duplicated(keep = "first"),:]

data.drop_duplicates(keep = "first")
data.drop_duplicates(keep = "last")
data.drop_duplicates(keep = False,inplace = True)  #by default inplace = False
data 

#Creating dummies

iris = pd.read_csv("C:\\Users\\Hp\\Desktop\\work\\Python\\Basics\\pandas\\iris.csv")
iris.head()

iris["setosa"] = iris.Species.map({"setosa" : 1,"versicolor":0, "virginica" : 0})
iris.head()

pd.get_dummies(iris.Species,prefix = "Species")
pd.get_dummies(iris.Species,prefix = "Species").iloc[:,0:1]  #1 is not included
species_dummies = pd.get_dummies(iris.Species,prefix = "Species").iloc[:,0:]


iris = pd.concat([iris,species_dummies],axis = 1)
iris.head()

pd.get_dummies(iris,columns = ["Species"],drop_first = True).head()

#Ranking
iris.rank() 

#Ranking by a specific variable

iris['Rank'] = iris.sort_values(['Sepal.Length'], ascending=[True]).groupby(['Species']).cumcount() + 1
iris.head( )

#Alternatively
iris['Rank2'] = iris['Sepal.Length'].groupby(iris["Species"]).rank(ascending=1)
iris.head() 


#Calculating the Cumulative sum
iris['cum_sum'] = iris["Sepal.Length"].cumsum()
iris.head()

#Cumulative sum by a variable
iris["cumsum2"] = iris.groupby(["Species"])["Sepal.Length"].cumsum()
iris.head()

#Calculating the percentiles.
iris.quantile(0.5)
iris.quantile([0.1,0.2,0.5])
iris.quantile(0.55)

#if else in Python

students = pd.DataFrame({'Names': ['John','Mary','Henry','Augustus','Kenny'],
                         'Zodiac Signs': ['Aquarius','Libra','Gemini','Pisces','Virgo']})

def name(row):
    if row["Names"] in ["John","Henry"]:
        return "yes"
    else:
        return "no"

students['flag'] = students.apply(name, axis=1)
students


import numpy as np
students['flag'] = np.where(students['Names'].isin(['John','Henry']), 'yes', 'no')
students

#Multiple Conditions : If Else-if Else
def mname(row):
    if row["Names"] == "John" and row["Zodiac Signs"] == "Aquarius" :
        return "yellow"
    elif row["Names"] == "Mary" and row["Zodiac Signs"] == "Libra" :
        return "blue"
    elif row["Zodiac Signs"] == "Pisces" :
        return "blue"
    else:
        return "black"

students['color'] = students.apply(mname, axis=1)
students

conditions = [
    (students['Names'] == 'John') & (students['Zodiac Signs'] == 'Aquarius'),
    (students['Names'] == 'Mary') & (students['Zodiac Signs'] == 'Libra'),
    (students['Zodiac Signs'] == 'Pisces')]
choices = ['yellow', 'blue', 'purple']
students['color'] = np.select(conditions, choices, default='black')
students


#Select numeric or categorical columns only

data1 = iris.select_dtypes(include=[np.number])
data1.head()

data3 = iris._get_numeric_data()
data3.head(3)

#For selecting categorical variables
data4 = iris.select_dtypes(include = ['object'])
data4.head(2)

#Concatenating

students = pd.DataFrame({'Names': ['John','Mary','Henry','Augustus','Kenny'],
                         'Zodiac Signs': ['Aquarius','Libra','Gemini','Pisces','Virgo']})
students2 = pd.DataFrame({'Names': ['John','Mary','Henry','Augustus','Kenny'],
                          'Marks' : [50,81,98,25,35]})
    
data = pd.concat([students,students2])  #by default axis = 0
data = pd.concat([students,students2],axis = 1)
data

students.append(students2)  #for rows
lasses = {'x': students, 'y': students2}
result = pd.concat(classes)
result


#Merging or joining on the basis of common variable.

students = pd.DataFrame({'Names': ['John','Mary','Henry','Maria'],
                         'Zodiac Signs': ['Aquarius','Libra','Gemini','Capricorn']})
students2 = pd.DataFrame({'Names': ['John','Mary','Henry','Augustus','Kenny'],
                          'Marks' : [50,81,98,25,35]})
    
result = pd.merge(students, students2, on='Names')  #it only takes intersections
result

result = pd.merge(students, students2, on='Names',how = "outer")  #it only takes unions
result 

result = pd.merge(students, students2, on='Names',how = "left")
result

result = pd.merge(students, students2, on='Names',how = "right",indicator = True)
result

import matplotlib
#Importing pyplot
from matplotlib import pyplot as plt

#Plotting to our canvas
plt.plot([1,2,3],[4,5,1])

#Showing what we plotted
plt.show()

x = [5,8,10]
y = [12,16,6]

plt.plot(x,y)

plt.title('Epic Info')
plt.ylabel('Y axis')
plt.xlabel('X axis')

plt.show()


#Plot Histogram
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_excel("E:/First.xlsx", "Sheet1")
#Plots in matplotlib reside within a figure object, use plt.figure to create new figure
fig=plt.figure()
#Create one or more subplots using add_subplot, because you can't create blank figure
ax = fig.add_subplot(1,1,1)
#Variable
ax.hist(df['Age'],bins = 5)
#Labels and Tit
plt.title('Age distribution')
plt.xlabel('Age')
plt.ylabel('#Employee')
plt.show()


#Scatter plot:
    
#Plots in matplotlib reside within a figure object, use plt.figure to create new figure fig=plt.figure()
#Create one or more subplots using add_subplot, because you can't create blank figure
ax = fig.add_subplot(1,1,1)
#Variable
ax.scatter(df['Age'],df['Sales'])
#Labels and Tit
plt.title('Sales and Age distribution')
plt.xlabel('Age')
plt.ylabel('Sales')
plt.show()


#Box-plot:

import seaborn as sns 
sns.boxplot(df['Age']) 
sns.despine()

#Convert character date to Date:

from datetime import datetime
char_date = 'Apr 1 2015 1:20 PM' #creating example character date
date_obj = datetime.strptime(char_date, '%b %d %Y %I:%M %p')
print date_obj  

import datetime

x = datetime.datetime.now()
print(x)
print(x.year)
print(x.strftime("%A"))
x = datetime.datetime(2020, 5, 17)
print(x) 
print(x.strftime("%B"))               
