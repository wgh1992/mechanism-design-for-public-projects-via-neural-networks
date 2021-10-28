#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.distributions as D
from torch.utils.data import DataLoader, TensorDataset
global  samplesJoint,tp_dataloader,tp_dataloader_testing,dp,decision,dp_H,decision_H
from pylab import *
from scipy.stats import beta
import random 
n = 5
epochs = 1
supervisionEpochs = 1
lr = 0.0001
log_interval = 5
trainSize = 40000#100000
percentage_train_test= 0.75
penaltyLambda = 10
doublePeakHighMean = 0.85
doublePeakLowMean = 0.15
doublePeakStd = 0.1
uniformlow=0
uniformhigh=1.0
normalloc = 0.2
normalscale = 0.1

cauchyloc = 1/n
cauchyscalen = 0.004

exponentialhigh = 15 #Symbol("b", real=True)
exponentiallow  = 15 #Symbol("a", real=True)

Sigmoidrate=0.00005
Sigmoidbias=0.001



independentnormalloc1=[(float(ii)+1)/(2*n+1) for ii in range(n,0,-1)]
independentnormalscale1=[0.05 for ii in range(n)]

independentnormalloc2=[(float(ii)+1)/(2*n+1) for ii in range(1,n+1,1)]
independentnormalscale2=[0.05 for ii in range(n)]
stage=["twopeak"]
# "twopeak","normal","uniform","independent1","independent2","cauchy","U-exponential"
order1name=["random initializing"]
numberofpeople=['0','1','2']
# "costsharing","dp","heuristic","random initializing"


# In[2]:


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

#d8 = beta(betahigh,betalow)
#d9 = D.beta.Beta(betahigh,betalow)
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
#     elif(y=="beta"):
#         return torch.tensor(d8.cdf(x));
    elif(y=="U-exponential"):
        return (d81.cdf(x) + (1.0 - d82.cdf(1.0-x)))  / 2 
print(cdf(0.5,"U-exponential"))

print(d81.cdf(0))


# In[3]:


def bitsToPayments(bits):
    if torch.sum(bits).item() == 0:
        return torch.ones(n)
    bits = bits.type(torch.float32)
    negBits = torch.ones(n) - bits
    payments = model(bits)
    payments = payments - 1000 * negBits
    payments = torch.softmax(payments, 0)
    payments = payments + negBits
    return payments

def tpToBits(tp, bits=torch.ones(n).type(torch.uint8)):
    payments = bitsToPayments(bits)
    newBits = (tp >= payments).type(torch.uint8)
    if torch.equal(newBits, bits):
        return bits
    else:
        return tpToBits(tp, newBits)#bits-bits#tpToBits(tp, newBits)

def tpToPayments(tp):
    return bitsToPayments(tpToBits(tp))


def tpToTotalDelay(tp):
    return n - torch.sum(tpToBits(tp).type(torch.float32))

def tpToBits0(tp, deep ,bits=torch.ones(n).type(torch.float32)):
    payments = bitsToPayments(bits)
    newBits = torch.sigmoid((tp - payments + Sigmoidbias)/Sigmoidrate)
    if torch.allclose(newBits, bits,rtol=0.01,atol=0.01) or deep>=n:
        return bits
    else:
        return tpToBits0(tp,deep+1 ,newBits)#bits-bits#tpToBits(tp, newBits)

def tpToTotalDelay0(tp):
    return n - torch.sum(tpToBits0(tp,0).type(torch.float32))


# In[4]:


def dpSupervisionRule(tp):
    global  dp,decision
    tp = list(tp.numpy())
    bits = [1 for ii in range(n)]
    payments = [0.0 for ii in range(n)]
    money = dpPrecision
    yes = 0
    for i in range(n):
        offerIndex = decision[n - i, money, yes]
        offer = float(offerIndex) / dpPrecision
        if tp[i] >= offer:
            money -= offerIndex
            yes += 1
            bits[i] = 1
            payments[i] = offer
        else:
            bits[i] = 0
            payments[i] = 1
    if money > 0:
        bits = [0 for ii in range(n)]
        payments = [1.0 for ii in range(n)]
    bits = torch.tensor(bits, dtype=torch.uint8)
    payments = torch.tensor(payments, dtype=torch.float32)
#     print()
#     print(tp)
#     print(bits)
#     print(payments)
#     print()
    return (bits, payments)


def heuristicSupervisionRule(tp):
    global  dp_H,decision_H
    tp = list(tp.numpy())
    bits = [1 for ii in range(n)]
    payments = [0.0 for ii in range(n)]
    tempold=-1;
    for turns in range(n,0,-1):
        money = dpPrecision
        j=0
        tempo=sum(bits)
        #print("bits",tempo)
        for i in range(n):
            if(j>=n):
                break;
            
            offerIndex = decision_H[tempo ,tempo - i, money]
            offer = float(offerIndex) / dpPrecision
            while(j<n):
                if(bits[j]!=0):
                    break;
                j+=1;
            if(j>=n):
                break;
            if tp[j] >= offer:
                #print(money,j,tp[j],offer)
                money -= offerIndex
                bits[j] = 1
                payments[j] = offer
            else:
                bits[j] = 0;
                payments[j] = 1.0;
            j+=1;
        #print("money",money)
        if(money==0 and tempold==tempo):
            break;
        tempold=tempo;
    if money > 0:
        bits = [0 for ii in range(n)]
        payments = [1.0 for ii in range(n)]

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
    for k in range(n, -1, -1):
        if k == 0:
            break
        bits = [1 if tp[ii] >= 1.0 / k else 0 for ii in range(n)]
        if sum(bits) == k:
            break
    if k == 0:
        payments = [1 for ii in range(n)]
    else:
        payments = [1.0 / k if bits[ii] == 1 else 1 for ii in range(n)]
    bits = torch.tensor(bits, dtype=torch.uint8)
    payments = torch.tensor(payments, dtype=torch.float32)
    return (bits, payments)


def costSharingDelay(tp):
    return float(n - torch.sum(costSharingSupervisionRule(tp)[0]).item())

def dpDelay(tp):
    return float(n - torch.sum(dpSupervisionRule(tp)[0]).item())

def heuristicDelay(tp):
    return float(n - torch.sum(heuristicSupervisionRule(tp)[0]).item())


# templire=0;
# num=1
# for i in range(num):
#     temp=torch.tensor(samplesJoint[i])
#     #print(temp)
#     #print(temp);
#     #print(heuristicSupervisionRule(temp))
#     #print(dpSupervisionRule(temp))
#     #print(dpDelay(temp))
#     res=dpDelay(temp)
#     templire+=res
#     #print("delay",res)
#     #print()
# print(templire/num)


# In[5]:


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
        for bitsMoreOnes in allBits:
            for i in range(n):
                if bitsMoreOnes[i] == 1:
                    bitsLessOnes = bitsMoreOnes.clone()
                    bitsLessOnes[i] = 0
                    
                    penalty = penalty + torch.sum(
                        torch.relu(
                            bitsToPayments(bitsMoreOnes) - bitsToPayments(bitsLessOnes)
                        )
                    )
        loss = penalty * penaltyLambda /100000
        
        for tp in tp_batch:
            bits, payments = supervisionRule(tp)
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


def train0(epoch):
    global losslisttemp; 
    model.train()
    for batch_idx, (tp_batch,) in enumerate(tp_dataloader):
        optimizer.zero_grad()
        penalty = 0
        for bitsMoreOnes in allBits:
            for i in range(n):
                if bitsMoreOnes[i] == 1:
                    bitsLessOnes = bitsMoreOnes.clone()
                    bitsLessOnes[i] = 0
                    penalty = penalty + torch.sum(
                        torch.relu(
                            bitsToPayments(bitsMoreOnes) - bitsToPayments(bitsLessOnes)
                        )
                    )
        loss = penalty * penaltyLambda
        for tp in tp_batch:
            delay0 = tpToTotalDelay0(tp)
            loss = loss + delay0

        loss = loss / len(tp_batch)
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
            
    print("penalty:",penalty.item())

def train1(epoch):
    global losslisttemp; 
    model.train()
    for batch_idx, (tp_batch,) in enumerate(tp_dataloader):
        optimizer.zero_grad()
        penalty = 0
        for bitsMoreOnes in allBits:
            for i in range(n):
                if bitsMoreOnes[i] == 1:
                    bitsLessOnes = bitsMoreOnes.clone()
                    bitsLessOnes[i] = 0
                    penalty = penalty + torch.sum(
                        torch.relu(
                            bitsToPayments(bitsMoreOnes) - bitsToPayments(bitsLessOnes)
                        )
                    )
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
            
    print("penalty:",penalty.item())
    
def train2(epoch):
    global losslisttemp; 
    model.train()
    for batch_idx, (tp_batch,) in enumerate(tp_dataloader):
        optimizer.zero_grad()
        penalty = 0
        for bitsMoreOnes in allBits:
            for i in range(n):
                if bitsMoreOnes[i] == 1:
                    bitsLessOnes = bitsMoreOnes.clone()
                    bitsLessOnes[i] = 0
                    penalty = penalty + torch.sum(
                        torch.relu(
                            bitsToPayments(bitsMoreOnes) - bitsToPayments(bitsLessOnes)
                        )
                    )
        loss = penalty * penaltyLambda
        for tp in tp_batch:
            for i in range(n):
                x1=random.randint(0,n-1);
                x2=random.randint(0,n-1);
                while(x2==x1):
                    x2=random.randint(0,n-1);
                    
                tp11 = tp.clone()
                tp11[x1] = 1
                tp11[x2] = 1
                tp10 = tp.clone()
                tp10[x1] = 1
                tp10[x2] = 0
                tp01 = tp.clone()
                tp01[x1] = 0
                tp01[x2] = 1
                tp00 = tp.clone()
                tp00[x1] = 0
                tp00[x2] = 0
                
                offer1 = tpToPayments(tp11)[x1]
                offer2 = tpToPayments(tp11)[x2]
                offer3 = tpToPayments(tp10)[x1]
                offer4 = tpToPayments(tp01)[x2]
                
                #print(tp)
                #print(offer1,offer2)
                
                delay11 = tpToTotalDelay(tp11)
                delay01 = tpToTotalDelay(tp01)
                delay10 = tpToTotalDelay(tp10)
                delay00 = tpToTotalDelay(tp00)
                if(order!="independent1" and order!="independent2"):
#                     print ((1.0-cdf(offer1,order)) * (1.0-cdf(offer2,order))+                                                   
#                            (1.0-cdf(offer3,order)) * cdf(offer2,order)+
#                            cdf(offer1,order) * (1.0-cdf(offer4,order))+
#                            (cdf(offer3,order) * cdf(offer4,order) -
#                                      (cdf(offer3,order)-cdf(offer1,order))*(cdf(offer4,order)-cdf(offer2,order)) )
#                           )
                    loss = loss +(      (1.0-cdf(offer1,order)) * (1.0-cdf(offer2,order)) * delay11 +
                                  (1.0-cdf(offer3,order)) * cdf(offer2,order) * delay10 +
                                  cdf(offer1,order) * (1.0-cdf(offer4,order)) * delay01 +
                                    (cdf(offer3,order) * cdf(offer4,order) -
                                     (cdf(offer3,order)-cdf(offer1,order))*(cdf(offer4,order)-cdf(offer2,order)) )* delay00 
                                                       )
                else:
                    loss = loss +(      (1.0-cdf(offer1,order,x1)) * (1.0-cdf(offer2,order),x2) * delay11 +
                                  (1.0-cdf(offer3,order,x1)) * cdf(offer2,order,x2) * delay10 +
                                  cdf(offer1,order,x1) * (1.0-cdf(offer4,order,x2)) * delay01 +
                                    (cdf(offer3,order,x1) * cdf(offer4,order,x2) -
                                     (cdf(offer3,order,x1)-cdf(offer1,order,x1))*(cdf(offer4,order,x2)-cdf(offer2,order,x2)) )* delay00 
                                                       )

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
            
    print("penalty:",penalty.item())
    
allBits = [torch.tensor(bits, dtype=torch.int16) for bits in itertools.product([0, 1], repeat=n)]

def test():
    global losslisttemp; 
    model.eval()

    with torch.no_grad():
        costSharingLoss = 0
        dpLoss = 0
        nnLoss = 0
        heuristicLoss=0
        lenLoss= 0
        for (tp_batch,) in tp_dataloader_testing:

            for tp in tp_batch:
                costSharingLoss += costSharingDelay(tp)
                dpLoss += dpDelay(tp)
                nnLoss += tpToTotalDelay(tp)
                heuristicLoss+= heuristicDelay(tp)
            lenLoss+=len(tp_batch)
        costSharingLoss /= lenLoss
        dpLoss /= lenLoss
        nnLoss /= lenLoss
        heuristicLoss /= lenLoss
        #print(lenLoss)
        losslisttemp.append(nnLoss);
        recordAndReport("NN", runningLossNN, nnLoss)
        recordAndReport("CS", runningLossCS, costSharingLoss)
        recordAndReport("DP", runningLossDP, dpLoss)
        recordAndReport("heuristic", runningLossHeuristic, heuristicLoss)
        print("DP:",dp[n, dpPrecision,0])
        #for i in range(n, 0, -1):
        #    print("Heuristic:",i,5*(1-dp_H[i, i, dpPrecision]))
        for i in range(n, 0, -1):
            print(
                    tpToPayments(
                            torch.tensor([0 if ii >= i else 1 for ii in range(n)], dtype=torch.float32)
                    )
                )
            
dpPrecision = 100 


# In[6]:


def producedata(order):
    global  samplesJoint,tp_dataloader,tp_dataloader_testing,dp,decision,dp_H,decision_H
    dpPrecision = 100 
    if(order=="twopeak"):
        print("loc",doublePeakLowMean, "scale",doublePeakStd)
        print("loc",doublePeakHighMean, "scale",doublePeakStd)
        signals = np.random.randint(2, size=(trainSize, n))
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
    elif(order=="normal"):
        print("loc",normalloc, "scale",normalscale)
        samples1 = np.random.normal(
            loc=normalloc, scale=normalscale, size=(trainSize, n)
        )
        for i in range(trainSize):
            for j in range(n):
                while samples1[i, j] < 0 or samples1[i, j] > 1:
                    samples1[i, j] = np.random.normal(
                        loc=normalloc, scale=normalscale
                    )
        samplesJoint = samples1
        tp_tensor = torch.tensor(samplesJoint, dtype=torch.float32)
    elif(order=="uniform"):  
        print("uniformlow",uniformlow, "uniformhigh",uniformhigh)
        samples1 = np.random.uniform(
            uniformlow, uniformhigh, size=(trainSize, n)
        )
        for i in range(trainSize):
            for j in range(n):
                while samples1[i, j] < 0 or samples1[i, j] > 1:
                    samples1[i, j] = np.random.normal(
                        uniformlow, uniformhigh
                    )
        samplesJoint = samples1
        tp_tensor = torch.tensor(samplesJoint, dtype=torch.float32)
    elif(order=="independent1"):
        print("loc",independentnormalloc1,"scale",independentnormalscale1)
        samples1 = np.random.uniform(
            uniformlow, uniformhigh, size=(trainSize, n)
        )
        for i in range(trainSize):
            for j in range(n):
                samples1[i, j] = np.random.normal(
                        independentnormalloc1[j], independentnormalscale1[j]
                    )
                while samples1[i, j] < 0 or samples1[i, j] > 1:
                    samples1[i, j] = np.random.normal(
                        independentnormalloc1[j], independentnormalscale1[j]
                    )
        samplesJoint = samples1
        tp_tensor = torch.tensor(samplesJoint, dtype=torch.float32)
    elif(order=="independent2"):
        print("loc",independentnormalloc2, "scale",independentnormalscale2)
        samples1 = np.random.uniform(
            uniformlow, uniformhigh, size=(trainSize, n)
        )
        for i in range(trainSize):
            for j in range(n):
                samples1[i, j] = np.random.normal(
                        independentnormalloc2[j], independentnormalscale2[j]
                    )
                while samples1[i, j] < 0 or samples1[i, j] > 1:
                    samples1[i, j] = np.random.normal(
                        independentnormalloc2[j], independentnormalscale2[j]
                    )
        samplesJoint = samples1
        tp_tensor = torch.tensor(samplesJoint, dtype=torch.float32)
    elif(order=="cauchy"):
        print("cauchyloc",cauchyloc, "cauchyscale",cauchyscalen)
        samples1 = np.random.uniform(
            uniformlow, uniformhigh, size=(trainSize, n)
        )
        for i in range(trainSize):
            for j in range(n):
                samples1[i, j] = d7.rsample(torch.Size([1]))
                while samples1[i, j] < 0 or samples1[i, j] > 1:
                    samples1[i, j] = d7.rsample(torch.Size([1]))
                    
        samplesJoint = samples1
        tp_tensor = torch.tensor(samplesJoint, dtype=torch.float32)
    elif(order=="beta"):
        print("betalow",betalow, "betahigh",betahigh)
        samples1 = np.random.uniform(
            uniformlow, uniformhigh, size=(trainSize, n)
        )
        for i in range(trainSize):
            for j in range(n):
                samples1[i, j] = beta.rvs(betahigh,betalow,  size = 1)
                while samples1[i, j] < 0 or samples1[i, j] > 1:
                    samples1[i, j] = beta.rvs(betahigh,betalow,  size = 1)
                    
        samplesJoint = samples1
        tp_tensor = torch.tensor(samplesJoint, dtype=torch.float32)
    elif(order=="U-exponential"):
        print("loc",doublePeakLowMean, "scale",doublePeakStd)
        print("loc",doublePeakHighMean, "scale",doublePeakStd)
        signals = np.random.randint(2, size=(trainSize, n))
        samples1 = d81.rsample(torch.Size([trainSize, n])).numpy()
        for i in range(trainSize):
            for j in range(n):
                while samples1[i, j] < 0 or samples1[i, j] > 1:
                    samples1[i, j] = d81.rsample(torch.Size([1])).numpy()
                    
        samples2 = d82.rsample(torch.Size([trainSize, n])).numpy()
        
        for i in range(trainSize):
            for j in range(n):
                while samples2[i, j] < 0 or samples2[i, j] > 1:
                    samples2[i, j] = d82.rsample(torch.Size([1])).numpy()
        samples2 = 1.0 - samples2
        samplesJoint = signals * samples1 - (signals - 1.0) * samples2
        tp_tensor = torch.tensor(samplesJoint, dtype=torch.float32)
        plt.hist(samplesJoint,bins=500)
        plt.show()
        # tp_tensor = torch.tensor(np.random.rand(10000, n), dtype=torch.float32)
        
        
    tp_dataset = TensorDataset(tp_tensor[: int(trainSize * percentage_train_test)])
    tp_dataset_testing = TensorDataset(tp_tensor[int(trainSize * (1.0-percentage_train_test)) :])
    tp_dataloader = DataLoader(tp_dataset, batch_size=128, shuffle=True)
    tp_dataloader_testing = DataLoader(tp_dataset_testing, batch_size=256, shuffle=False)
    
    # howManyPpl left, money left, yes already
    dp = np.zeros([n + 1, dpPrecision + 1, n + 1])
    decision = np.zeros([n + 1, dpPrecision + 1, n + 1], dtype=np.uint8)
    # ppl = 0 left
    for yes in range(n + 1):
        for money in range(dpPrecision + 1):
            if money == 0:
                dp[0, 0, yes] = 0
            else:
                dp[0, money, yes] = yes# + 1.0
    for ppl in range(1,  n + 1):
        for yes in range(n + 1):
            for money in range(dpPrecision + 1):
                minSoFar = 1000000
                for offerIndex in range(money + 1):
                    offer = float(offerIndex) / dpPrecision
                    if(order!="independent1" and order!="independent2"):
                        res = (1 - cdf(offer,order)) * dp[
                        ppl - 1, money - offerIndex, min(yes + 1, n)
                        ] + cdf(offer,order) * (1 + dp[ppl - 1, money, yes])
                    else:
                        res = (1 - cdf(offer,order,n-ppl)) * dp[
                        ppl - 1, money - offerIndex, min(yes + 1, n)
                        ] + cdf(offer,order,n-ppl) * (1 + dp[ppl - 1, money, yes])
                    if minSoFar > res.item():
                        minSoFar = res.item()
                        decision[ppl, money, yes] = offerIndex
                dp[ppl, money, yes] = minSoFar
    
    print("dp",dp[n, dpPrecision, 0])
    
    # howManyPpl left, money left, yes already
    dp_H = np.zeros([n + 1 , n + 1, dpPrecision + 1])
    decision_H = np.zeros([n + 1 , n + 1, dpPrecision + 1], dtype=np.uint8)
    # ppl = 0 left
    for i in range(1,n+1):
        for money in range(dpPrecision + 1):
            if money == 0:
                dp_H[i, 0, 0] = 1
            else:
                offer = money / dpPrecision
                dp_H[i, 0, money] = 0#cdf(offer)# + 1.0
    for i in range(1,n+1):
        for ppl in range(1, i + 1):
            for money in range(dpPrecision + 1):
                maxSoFar = -1000000
                for offerIndex in range(money + 1):
                    offer = float(offerIndex) / dpPrecision
                    if(order!="independent1" and order!="independent2"):
                        res = (1-cdf(offer,order)) * dp_H[
                         i, ppl - 1, money - offerIndex
                        ]
                    else:
                        res = (1-cdf(offer,order,ppl-1)) * dp_H[
                         i, ppl - 1, money - offerIndex
                        ]
                    if maxSoFar < res.item():
                        maxSoFar = res.item()
                        decision_H[i, ppl, money] = offerIndex
                dp_H[i, ppl, money] = maxSoFar
                


# In[7]:


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        #m.bias.data.fill_(0.1)
    
for order in stage:
    producedata(order)




                
    #running
    for order1 in order1name:
        print("Supervised Aim:",order,order1)
        for trainingnumber in numberofpeople:
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

            runningLossNN = []
            runningLossCS = []
            runningLossDP = []
            runningLossHeuristic = []
            #model=torch.load("save/pytorchNN=5dp1");
            #model.eval()
            ##order1name=["costsharing","dp","heuristic","random initializing"]
            for epoch in range(1, supervisionEpochs + 1):
        #             print("distributionRatio",distributionRatio)
                if(order1=="costsharing"):
                    supervisionTrain(epoch, costSharingSupervisionRule)
                elif(order1=="dp"):
                    supervisionTrain(epoch, dpSupervisionRule)
                elif(order1=="heuristic"):
                    supervisionTrain(epoch, heuristicSupervisionRule)
                elif(order1=="random initializing"):
                    print("do nothing");

            test()

            for epoch in range(1, epochs + 1):
                if(trainingnumber=='0'):
                    train0(epoch)
                if(trainingnumber=='1'):
                    train1(epoch)
                if(trainingnumber=='2'):
                    train2(epoch)
                test()
            losslistname.append(order+" "+order1+" choose people:"+trainingnumber);
            losslist.append(losslisttemp);
            losslisttemp=[];
            savepath="save/pytorchNN=5all-phoenix"+order+str(order1)+trainingnumber
            torch.save(model, savepath);
            print("end")
            print()


# In[8]:


colorlist=["#D2691E",'#4169E1',"#9ACD32","#B22222","#FF00FF","#708090"]
for i in range(len(losslist)):
    plt.plot(losslist[i], '.-', color=colorlist[i], alpha=0.8, label=losslistname[i])

# 显示标签，如果不加这句，即使加了label='一些数字'的参数，最终还是不会显示标签
plt.legend(loc="best")
plt.xlabel('trianing times')
plt.ylabel('delay')
plt.title("n="+str(n))

plt.show()

