
# coding: utf-8

# In[1]:


# Required Packages
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model


# In[2]:


#Understanding train_test_split
from sklearn.cross_validation import train_test_split
X, y = np.arange(50).reshape((10,5)),range(10)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.5,random_state=0)
print("X VALUES:")
print(X)
print("X_train Data:")
print(X_train)
print("X_test Data:")
print(X_test)
print("y_train Data:")
print(y_train)
print("y_test Data:")
print(y_test)
print("y values:")
train_test_split(y)


# In[3]:


#Importing data directly from website
#import pandas as pd
#dataset_url = 'http://www.dsi.uminho.pt/~pcortez/forestfires/forestfires.csv'
#data = pd.read_csv(dataset_url)
#data.head()


# In[4]:


# Function to get data
def get_data(file_name):
    data = pd.read_csv(file_name)
    x_parameter = []
    y_parameter = []
    for single_square_feet ,single_price_value in zip(data['square_feet'],data['price']):
        x_parameter.append([float(single_square_feet)])
        y_parameter.append(float(single_price_value))
    return x_parameter,y_parameter


# In[5]:


#Printing data
x,y = get_data("C:/Users/Amit Kumar Mitra/Desktop/Downloads/House.csv")
print (x)
print(type(x))
print (y)


# In[6]:


# Function for Fitting data to Linear model
def linear_model_main(X_parameters,Y_parameters,predict_value):

    # Create linear regression object
    regr = linear_model.LinearRegression()
    regr.fit(X_parameters, Y_parameters)
    predict_outcome = regr.predict(predict_value)
    predictions = {}
    predictions['intercept'] = regr.intercept_
    predictions['coefficient'] = regr.coef_
    predictions['predicted_value'] = predict_outcome
    print("Outcome",predict_outcome)
    fig=plt.figure()
    plt.plot(predict_outcome)
    return predictions


# In[7]:


x1=np.transpose(x)
x.flatten()
#plt.bar(x,y)


# In[ ]:


x,y = get_data("C:/Users/Amit Kumar Mitra/Desktop/Downloads/House.csv")
predict_value=int(input("Enter the Area(Square Feet) for which you want to know Price: "))
result = linear_model_main(x,y,predict_value)
print ("Intercept value " , result['intercept'])
print ("coefficient" , result['coefficient'])
print ("Predicted value: ",result['predicted_value'])

# SELF
#pred=result['predicted_value']
#print(pred.dtype)
#pred1=pred.tolist()
#print(pred1)
#print(type(pred1))
#y.append(pred1[0])
#print(y)


# In[ ]:


# SELF
#predval=[]
#predval.append(predict_value)
#print(type(predval))
#x.append(predval)
#print(x)


# In[8]:


# Function to show the resutls of linear fit model
def show_linear_line(X_parameters,Y_parameters):
    # Create linear regression object
    regr = linear_model.LinearRegression()
    regr.fit(X_parameters, Y_parameters)
    plt.scatter(X_parameters,Y_parameters,color='blue')
    plt.plot(X_parameters,regr.predict(X_parameters),color='red',linewidth=4)
    plt.xticks(())
    plt.yticks(())
    plt.show()

print("Plottting Linear regression line")
show_linear_line(x,y)
print("Plottting the Values")
fig=plt.figure()
plt.plot(x,y,'+m')

