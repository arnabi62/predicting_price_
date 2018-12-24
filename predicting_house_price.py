
# coding: utf-8

# In[1]:


import pandas as pd
file=pd.read_csv('C:/Users/Amit Kumar Mitra/Desktop/Downloads/House.csv')
file.dropna()


# In[3]:


z=file['square_feet'].tolist()
y=file['price'].tolist()
import matplotlib.pyplot as plt
plt.plot(z,y,'*')
plt.xlabel('squarefeet')
plt.ylabel('price')
plt.show


# In[4]:


m=(y[1]-y[0])/(z[1]-z[0])
#print(m)
c=y[2]-m*z[2]
#print(c)
m1=(y[28]-y[27])/(z[28]-z[27])
#print(m1)
c1=y[29]-m*z[29]
#x=input("data")
print(m1)
print(c1)
#if(int(x)<=800):
   # k=m*int(x)+c
   # print(k)
#if(int(x)>800):
   # j=m1*int(x)+c1
   # print(j)


# In[ ]:





# In[6]:


m=(y[1]-y[0])/(z[1]-z[0])
#print(m)
c=y[2]-m*z[2]
#print(c)
m1=(y[28]-y[27])/(z[28]-z[27])
#print(m1)
c1=y[29]-m1*z[29]
x=input("data")
#print(m1)
#print(c1)
if(int(x)<=800):
    k=m*int(x)+c
    print(k)
if(int(x)>800):
    j=m1*int(x)+c1
    print(j)


# In[12]:


b=[]
import numpy as np
red=np.array(z)

for i in range(len(z)-1):
    k=red[i+1]-red[i]
    b.append(k)
    
j=np.array(b)    
#print(j)
k=np.unique(j)
array_size=len(k)
sum=0
for i in range(0, array_size):
    t=k[i]
    var=b.count(t)
    sum=sum+k[i]*var
mean=sum/len(z)

step_size=500/mean
print(step_size)
result=step_size*int(x)
print(result)
#b.count(25)


# In[15]:


median=(26*m+4*m1)/30
con=(26*c+4*c1)/30
x=input('enter the value')
result=median*int(x)+con
print(result)


# In[ ]:




