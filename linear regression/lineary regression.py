#!/usr/bin/env python
# coding: utf-8

# In[20]:


#下载库时利用pip install
from PIL import Image
import numpy as np

#保存图片路径
path="/Users/linjiaxi/Desktop/qq.JPG" 

#形成像素矩阵并且输出
im=Image.open(path)
mat=np.array(im)
print(mat)

#显示图片
import matplotlib.pyplot as plt
plt.imshow(im)
plt.axis('off')
plt.show()


# In[37]:


import numpy as np
import pandas as pd

#原始数据集
path="/Users/linjiaxi/Desktop/linear regression/day.csv"

#读取数据
bike=pd.read_csv(path,sep=",")
bike.head()


# In[38]:


#提取特征
features=['season','holiday','weekday','workingday','temp','atemp','hum','windspeed']
label='cnt'

X_train=bike[features]
Y_train=bike[label]

print('X_train shape:',X_train.shape)
print('Y_train shape:',Y_train.shape)


# In[39]:


#计算特征之间的相关性（对称矩阵）
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

corr_res = X_train.corr()
plt.subplots(figsize=(9,9))
sns.heatmap(corr_res, annot=True, vmax=1, square=True, cmap='coolwarm')
plt.show()


# In[40]:


#判断正态分布
from scipy import stats

def checkNORM(se,p=0.05,alt='two-sided',if_plot=True):
    print('数据量:',len(se))
    
    u = se.mean()  # 计算均值
    std=se.std()  # 计算标准差
    res=stats.kstest(rvs=se, cdf='norm', args=(u, std), alternative=alt)
    print('p值为',res[1])
    if res[1]>p:
        print('p值>',p,'符合正态分布')
    else:
        print('p值<=',p,'不符合正态分布')
   
    if if_plot==True:
        fig = plt.figure(figsize=(10,6))         
        se.hist(bins=30,alpha =0.5) #直方图 alpha表示透明度
        se.plot(kind ='kde', secondary_y=True) #核密度估计KDE
        plt.show()
 
    return res
res=checkNORM(Y_train)

from statsmodels.regression.linear_model import OLS,GLS #Ordinary least squares普通最小二乘法
import statsmodels.formula.api as smf

data_cols = ['season','holiday','weekday','workingday','weathersit','temp','atemp','hum','windspeed','cnt']
print(data_cols)

model = smf.ols(formula = 'cnt ~ season + holiday + weekday + workingday + weathersit + temp + atemp+ hum + windspeed ', data = bike[data_cols])
results= model.fit()
results.summary()


# In[15]:


#使用mindspore进行先行回归分析
import os
# os.environ['DEVICE_ID'] = '0'
import numpy as np

import mindspore as ms
from mindspore import nn
from mindspore import context

#context.set_context(mode=context.GRAPH_MODE, device_target="Ascend") #使用昇腾算力
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")      #使用本地CPU算力


# In[16]:


x = np.arange(-5, 5, 0.3)[:32].reshape((32, 1))
y = -5 * x +  0.1 * np.random.normal(loc=0.0, scale=20.0, size=x.shape)


# In[17]:


net = nn.Dense(1, 1)
loss_fn = nn.loss.MSELoss()
opt = nn.optim.SGD(net.trainable_params(), learning_rate=0.01)
with_loss = nn.WithLossCell(net, loss_fn)
train_step = nn.TrainOneStepCell(with_loss, opt).set_train()


# In[18]:


#利用epoch（扰动）
for epoch in range(2):
    loss = train_step(ms.Tensor(x, ms.float32), ms.Tensor(y, ms.float32))
    print('epoch: {0}, loss is {1}'.format(epoch, loss))


# In[19]:


wb = [x.asnumpy() for x in net.trainable_params()]
w, b = np.squeeze(wb[0]), np.squeeze(wb[1])
print('The true linear function is y = -5 * x + 0.1')
print('The trained linear model is y = {0} * x + {1}'.format(w, b))

for i in range(-10, 11, 5):
    print('x = {0}, predicted y = {1}'.format(i, net(ms.Tensor([[i]], ms.float32))))


# In[20]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(x, y, label='Samples')
plt.plot(x, w * x +  b, c='r', label='True function')
plt.plot(x, -5 * x +  0.1, c='b', label='Trained model')
plt.legend()


# In[ ]:





# In[ ]:




