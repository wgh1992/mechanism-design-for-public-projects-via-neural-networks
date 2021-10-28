#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from random import randint
import random
import torch.distributions as D
from torch.utils.data import DataLoader, TensorDataset
global  samplesJoint,tp_dataloader,tp_dataloader_testing,dp,decision,dp_H,decision_H
from pylab import *
from scipy.stats import beta
import math
n = 3
epochs = 1
supervisionEpochs = 1
lr = 0.0005
log_interval = 20
trainSize = 60000#100000
percentage_train_test= 0.25
penaltyLambda = 10
doublePeakHighMean = 0.9
doublePeakLowMean = 0.1
doublePeakStd = 0.1
uniformlow=0
uniformhigh=1.0
normalloc = 0.2
normalscale = 0.1

cauchyloc = 1/n
cauchyscalen = 0.004

exponentialhigh = 15 #Symbol("b", real=True)
exponentiallow  = 15 #Symbol("a", real=True)


beta_a = 0.1
beta_b  = 0.1
kumaraswamy_a = beta_a 
kumaraswamy_b = (1.0+(beta_a-1.0)*math.pow( (beta_a+beta_b-2.0)/(beta_a-1.0), beta_a) )/beta_a 
print(kumaraswamy_b)

independentnormalloc1=[(float(ii)+1)/(2*n+1) for ii in range(n,0,-1)]
independentnormalscale1=[0.05 for ii in range(n)]

independentnormalloc2=[(float(ii)+1)/(2*n+1) for ii in range(1,n+1,1)]
independentnormalscale2=[0.05 for ii in range(n)]
stage=["twopeak"]
order="twopeak"
# "twopeak","normal","uniform","independent1","independent2","cauchy","beta","U-exponential","arcsine"
order1name=["costsharing","random initializing","dp"]
#order1name=["random initializing1","random initializing2","random initializing3"]
# "costsharing","dp","heuristic","random initializing"


# In[2]:


# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")


# In[3]:


d1 = D.normal.Normal(doublePeakLowMean, doublePeakStd)
d2 = D.normal.Normal(doublePeakHighMean, doublePeakStd)
distributionRatio1 = (d1.cdf(1) + d2.cdf(1) - d1.cdf(0) - d2.cdf(0)) / 2
distributionBase1 = d1.cdf(0) + d2.cdf(0)

d3 = D.normal.Normal(normalloc, normalscale)
distributionRatio3 = d3.cdf(1) - d3.cdf(0)
distributionBase3 = d3.cdf(0)

d4 = D.uniform.Uniform(uniformlow,uniformhigh)
distributionRatio4 = d4.cdf(1) - d4.cdf(0)
distributionBase4 = d4.cdf(0)

d5 = [D.normal.Normal(independentnormalloc1[ii], independentnormalscale1[ii]) for ii in range(n)]
d6 = [D.normal.Normal(independentnormalloc2[ii], independentnormalscale2[ii]) for ii in range(n)]

d7 = D.cauchy.Cauchy(cauchyloc,cauchyscalen)

d81 = D.exponential.Exponential(exponentiallow)
d82 = D.exponential.Exponential(exponentialhigh)

d9 = D.beta.Beta(beta_a,beta_b)
#sample_d9=d9.rsample(torch.Size([100000]))

#d9_sample=np.linspace(0.0001, 0.9999, 10000)
#d9_pdf=torch.exp(d9.log_prob(torch.tensor(d9_sample,dtype=torch.float32)))
#d9_delta=d9_sample[1]-d9_sample[0]
#d9_sum_pdf=torch.sum(d9_pdf*d9_delta)

d10 = D.beta.Beta(0.5,0.5)

def cdf(x,y, i=None):
    if(y=="twopeak"):
        return (d1.cdf(x) + d2.cdf(x) - distributionBase1) / 2 / distributionRatio1
    elif(y=="normal"):
        return (d3.cdf(x)-distributionBase3)/distributionRatio3;
    elif(y=="uniform"):
        return (d4.cdf(x)-distributionBase4)/distributionRatio4;
    elif(y=="independent1"):
        return d5[i].cdf(x);
    elif(y=="independent2"):
        return d6[i].cdf(x);
    elif(y=="cauchy"):
        return d7.cdf(x);
    elif(y=="beta"):
#         sum_cdf=0.0;
#         if(x<0.0001):
#             x=0.00011;
#         if(x>0.9999):
#             x=0.99989;
#         for i in range(len(d9_pdf)):
#             if(d9_sample[i]<x):
#                 sum_cdf+=d9_pdf[i]*d9_delta;
#             else:
#                 sum_cdf+=(d9_pdf[i]+d9_pdf[i-1])/ 2 *(x-d9_sample[i-1])
#                 break;
#         return sum_cdf/d9_sum_pdf
#         cdf_v=torch.sum((sample_d9<(x)), dtype=torch.float32)/100000
#         return cdf_v
#    F(x|a,b)=1–(1–x^a)^b
        if(x<0.0000001):
            x=0.0000001
        elif(x >0.9999999):
            x=0.9999999
        try:
            return 1.0-torch.pow(1.0-torch.pow(x,kumaraswamy_a),kumaraswamy_b);
        except:
            return 1.0-torch.pow(1.0-torch.pow(torch.tensor(x,dtype=torch.float32),kumaraswamy_a),kumaraswamy_b);
    elif(y=="arcsine"):
        #
        if(x<0.0000001):
            x=0.0000001
        elif(x >0.9999999):
            x=0.9999999
        try:
            res=2.0/math.pi * torch.asin(torch.sqrt(x))
            #print(x)
            return res# + 0.0001*1.0/(
            #math.pi * torch.sqrt(torch.tensor(x)*torch.tensor(1.0-x)))
        except:
            return 2.0/math.pi * torch.asin(torch.sqrt(torch.tensor(x,dtype=torch.float32)))# + 0.0001*1.0/(
            #math.pi * torch.sqrt(torch.tensor(x)*torch.tensor(1.0-x)))
    elif(y=="U-exponential"):
        return (d81.cdf(x) + (1.0 - d82.cdf(1.0-x)))  / 2 
    


# In[4]:


dpPrecision = 100
# howManyPpl left, money left, yes already
dp = np.zeros([n + 1, dpPrecision + 1])
decision = np.zeros([n + 1, dpPrecision + 1], dtype=np.uint8)
# ppl = 0 left
for money in range(dpPrecision + 1):
    if money == 0:
        dp[0, 0] = 1
    else:
        offer = money / dpPrecision
        dp[0, money] = 0#cdf(offer)# + 1.0
for ppl in range(1, n + 1):
    for money in range(dpPrecision + 1):
        maxSoFar = -1_000_000
        for offerIndex in range(money + 1):
            offer = offerIndex / dpPrecision
            res = (1-cdf(offer,order)) * dp[
                 ppl - 1, money - offerIndex
                ]
            if maxSoFar < res:
                maxSoFar = res
                decision[ppl, money] = offerIndex
        dp[ppl, money] = maxSoFar
print(dp[n, dpPrecision])
print(n*(1-dp[n, dpPrecision]))


# In[5]:



        
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.1)


# In[6]:


signals = np.random.randint(2, size=(trainSize, n))
# samples1 = np.random.uniform(low=0.1, high=0.2, size=(trainSize, n))
# samples2 = np.random.uniform(low=0.8, high=0.9, size=(trainSize, n))
samples1 = np.random.normal(
    loc=doublePeakLowMean, scale=doublePeakStd, size=(trainSize, n)
)
for i in range(trainSize):
    for j in range(n):
        while samples1[i, j] < 0 or samples1[i, j] > 1:
            samples1[i, j] = np.random.normal(
                loc=doublePeakLowMean, scale=doublePeakStd
            )
samples2 = np.random.normal(
    loc=doublePeakHighMean, scale=doublePeakStd, size=(trainSize, n)
)
for i in range(trainSize):
    for j in range(n):
        while samples2[i, j] < 0 or samples2[i, j] > 1:
            samples2[i, j] = np.random.normal(
                loc=doublePeakHighMean, scale=doublePeakStd
            )
samplesJoint = signals * samples1 - (signals - 1) * samples2
tp_tensor = torch.tensor(samplesJoint, dtype=torch.float32)
# tp_tensor = torch.tensor(np.random.rand(10000, n), dtype=torch.float32)
tp_dataset = TensorDataset(tp_tensor[: int(trainSize * percentage_train_test)])
tp_dataset_testing = TensorDataset(tp_tensor[int(trainSize * (1.0-percentage_train_test)) :])
tp_dataloader = DataLoader(tp_dataset, batch_size=128, shuffle=True)
tp_dataloader_testing = DataLoader(tp_dataset_testing, batch_size=256, shuffle=False)
plt.hist(samplesJoint,bins=500)
plt.show()
runningLossNN = []
runningLossCS = []
runningLossDP = []
# for mapping binary to payments before softmax
model = nn.Sequential(
    nn.Linear(n, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, n),
)

model.apply(init_weights)
# optimizer = optim.SGD(model.parameters(), lr=lr)
optimizer = optim.Adam(model.parameters(), lr=lr)


# In[7]:


def bitsToPayments(bits):
    bits = bits.type(torch.float32)
    payments = model(torch.ones(n))
    payments = torch.softmax(payments, 0)
    return payments


def tpToBits(tp, bits=torch.ones(n).type(torch.uint8)):
    payments = bitsToPayments(bits)
    newBits = (tp >= payments).type(torch.uint8)
    
    if torch.equal(newBits, bits):
        return bits
    else:
        return bits-bits#tpToBits(tp, newBits)


def tpToPayments(tp):
    return bitsToPayments(tpToBits(tp))


def tpToTotalDelay(tp):
    if torch.sum(tpToBits(tp).type(torch.float32))==n:
        return 0
    else:
        return n


# In[8]:


def plan_dp(temp,debug):
    if(debug==1):
        print(temp)
    remain=dpPrecision
    ans =0;
    o_list=[];
    remain_list=[];
    for ppl in range(n,0,-1):
        o=decision[ppl, remain]
        if(debug==1):
            print(o,remain)
        o_list.append(o)
        remain_list.append(remain);
        if(o<temp[n-ppl]):
            remain-=int(o);
        elif (remain>0):
            ans=n;
    if(remain<=1):
        return ans,o_list;
    else:
        return n,o_list;
    


# In[9]:


ans_list=[];
for i in range(5):
    temp=samplesJoint[i]*dpPrecision
    #print(temp)
    tempres=plan_dp(temp,1)
    ans_list.append(tempres[0]);
    print(tempres)
    #print("\n",temp)
    #print(plan_dp(temp,1))
for i in range(10000):
    temp=samplesJoint[i]*dpPrecision
    #print(temp)
    ans_list.append(plan_dp(temp,0)[0]);
    #print("\n",temp)
    #print(plan_dp(temp))
print(sum(ans_list)/len(ans_list))


# In[10]:


def dpSupervisionRule(tp):
    tp = list(tp.numpy())
    bits = [1 for ii in range(n)]
    payments = [0 for ii in range(n)]
    money = dpPrecision
    yes = 0
    for i in range(n):
        offerIndex = decision[n - i, money]
        offer = offerIndex / dpPrecision
        if tp[i] >= offer:
            money -= offerIndex
            yes += 1
            bits[i] = 1
            payments[i] = offer
        else:
            bits = [0 for ii in range(n)]
            payments = [1 for ii in range(n)]
            money=1
            #bits[i] = 0
            #payments[i] = 0#1
            break
    if money > 0:
        bits = [0 for ii in range(n)]
        payments = [1 for ii in range(n)]

    bits = torch.tensor(bits, dtype=torch.float32)
    payments = torch.tensor(payments, dtype=torch.float32)
    # print()
    # print(tp)
    # print(bits)
    # print(payments)
    # print()
    return (bits, payments)




def costSharingSupervisionRule(tp):
    tp = list(tp.numpy())
    #for k in range(n, -1, -1):
    k=n;
    bits = [1 if tp[ii] >= 1 / k else 0 for ii in range(n)]
    payments = [1 / k  for ii in range(n)]
        
    bits = torch.tensor(bits, dtype=torch.uint8)
    payments = torch.tensor(payments, dtype=torch.float32)
    return (bits, payments)


def costSharingDelay(tp):
    if torch.sum(costSharingSupervisionRule(tp)[0]).item() == n:
        return 0
    else:
        return n;

def dpDelay(tp):
    if torch.sum(dpSupervisionRule(tp)[0]).item() == n:
        return 0
    else:
        return n;


# In[11]:


for batch_idx, (tp_batch,) in enumerate(tp_dataloader_testing):
    print(tp_batch[0])
    print(costSharingDelay(tp_batch[0]))
    print(costSharingSupervisionRule(tp_batch[0]))
    print(torch.sum(costSharingSupervisionRule(tp_batch[0])[0]).item())
    print("\n")


# In[12]:


allBits = [torch.tensor(bits) for bits in itertools.product([0, 1], repeat=n)]
print(allBits)

for batch_idx, (tp_batch,) in enumerate(tp_dataloader_testing):
    penalty = 0
    loss = penalty * penaltyLambda
    for tp in tp_batch:
            for i in range(n):
                tp1 = tp.clone()
                tp1[i] = 1
                tp0 = tp.clone()
                tp0[i] = 0
                offer = tpToPayments(tp1)[i]
                delay1 = tpToTotalDelay(tp1)
                delay0 = tpToTotalDelay(tp0)
                #loss = loss + (1 - cdf(offer)) * delay1 + cdf(offer) * delay0
                loss = loss + (1 - cdf(offer,order)) * delay1 + cdf(offer,order) * delay0
    print()
    print(tp)
    tp1 = tp.clone()
    tp1[0] = 1
    tp0 = tp.clone()
    tp0[0] = 0
    offer = tpToPayments(tp1)[0]
    print(offer)
    print(tpToPayments(tp1))
    print(delay1)
    print(delay0)
    break
#print(loss)
#print(penalty)


# In[13]:


def recordAndReport(name, source, loss):
    source.append(loss)
    realLength = len(source)
    #print(f"{name} ({realLength}): {loss}")
    print(name,realLength,":" ,loss)

def supervisionTrain(epoch, supervisionRule):
    model.train()
    for batch_idx, (tp_batch,) in enumerate(tp_dataloader):
        optimizer.zero_grad()
        penalty = 0
        loss = penalty * penaltyLambda /100000
        
        for tp in tp_batch:
            bits, payments = supervisionRule(tp)
            #print("bits",bitsToPayments(bits))
            loss = loss + F.mse_loss(bitsToPayments(bits), payments)

        loss = loss / len(tp_batch)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(tp_batch),
                    len(tp_dataloader.dataset),
                    100.0 * batch_idx / len(tp_dataloader),
                    loss.item(),
                )
            )

losslist=[];
losslistname=[];
losslisttemp=[];
def test_batch():
    global losslisttemp; 
    model.eval()
    with torch.no_grad():
        lenLoss= 0
        nnLoss = 0
        for (tp_batch,) in tp_dataloader_testing:
            for tp in tp_batch:
                nnLoss += tpToTotalDelay(tp)
            lenLoss+=len(tp_batch)
        nnLoss/=lenLoss
    return nnLoss


def train(epoch):
    global losslisttemp; 
    model.train()
    for batch_idx, (tp_batch,) in enumerate(tp_dataloader):
        optimizer.zero_grad()
        penalty = 0
        loss = penalty * penaltyLambda
        for tp in tp_batch:
            for i in range(n):
                tp1 = tp.clone()
                tp1[i] = 1
                tp0 = tp.clone()
                tp0[i] = 0
                offer = tpToPayments(tp1)[i]
                delay1 = tpToTotalDelay(tp1)
                delay0 = tpToTotalDelay(tp0)
                if(order!="independent1" and order!="independent2"):
                    loss = loss + (1 - cdf(offer,order)) * delay1 + cdf(offer,order) * delay0
                else:
                    loss = loss + (1 - cdf(offer,order,i)) * delay1 + cdf(offer,order,i) * delay0

        loss = loss / len(tp_batch) / n
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            losstemp=test_batch();
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(tp_batch),
                    len(tp_dataloader.dataset),
                    100.0 * batch_idx / len(tp_dataloader),
                    loss.item(),
                ),"testing loss:",losstemp
            )
            losslisttemp.append(losstemp);
            
    
allBits = [torch.tensor(bits, dtype=torch.int16) for bits in itertools.product([0, 1], repeat=n)]

def test():
    global losslisttemp; 
    model.eval()

    with torch.no_grad():
        costSharingLoss = 0
        dpLoss = 0
        nnLoss = 0
        lenLoss= 0
        for (tp_batch,) in tp_dataloader_testing:

            for tp in tp_batch:
                costSharingLoss += costSharingDelay(tp)
                dpLoss += dpDelay(tp)
                nnLoss += tpToTotalDelay(tp)
            lenLoss+=len(tp_batch)
        costSharingLoss /= lenLoss
        dpLoss /= lenLoss
        nnLoss /= lenLoss
        #print(lenLoss)
        losslisttemp.append(nnLoss);
        recordAndReport("NN", runningLossNN, nnLoss)
        recordAndReport("CS", runningLossCS, costSharingLoss)
        recordAndReport("DP", runningLossDP, dpLoss)
        print("DP:",n*(1-dp[n, dpPrecision]))
        #for i in range(n, 0, -1):
        #    print("Heuristic:",i,5*(1-dp_H[i, i, dpPrecision]))
        for i in range(n, 0, -1):
            print(
                    tpToPayments(
                            torch.tensor([0 if ii >= i else 1 for ii in range(n)], dtype=torch.float32)
                    )
                )
            
dpPrecision = 100 


# In[14]:


for epoch in range(1, supervisionEpochs + 1):
    print(distributionRatio1)
    supervisionTrain(epoch, costSharingSupervisionRule)
    #supervisionTrain(epoch, dpSupervisionRule)
test()
for epoch in range(1, epochs + 1):
    
    train(epoch)
    test()


# In[15]:


order1=""
losslistname.append(order+" "+order1);
losslist.append(losslisttemp);
losslisttemp=[];


# In[16]:


colorlist=["#D2691E",'#4169E1',"#9ACD32","#B22222","#FF00FF","#708090"]
for i in range(len(losslist)):
    plt.plot(losslist[i], 'ro-', color=colorlist[i], alpha=0.8, label=losslistname[i])

# 显示标签，如果不加这句，即使加了label='一些数字'的参数，最终还是不会显示标签
plt.legend(loc="best")
plt.xlabel('trianing times')
plt.ylabel('delay')

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




