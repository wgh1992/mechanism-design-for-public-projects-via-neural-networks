import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import torch.distributions as D
from torch.utils.data import DataLoader, TensorDataset
from pylab import *
from scipy.stats import beta
import math

n=0
def inital(nn):
    global n,doublePeakHighMean,doublePeakLowMean,doublePeakStd,uniformlow,uniformhigh,normalloc,normalscale,cauchyloc,cauchyscalen,exponentialhigh,exponentiallow,dpPrecision,beta_a,beta_b
    global kumaraswamy_a,kumaraswamy_b,independentnormalloc1,independentnormalloc2,independentnormalscale1,independentnormalscale2
    global distributionBase1,distributionBase2,distributionBase3,distributionBase4
    global distributionRatio1,distributionRatio2,distributionRatio3,distributionRatio4
    global d1,d2,d3,d4,d5,d6,d7,d81,d82,d9,d10
    n=nn
    doublePeakHighMean = 0.85
    doublePeakLowMean = 0.15
    doublePeakStd = 0.1
    uniformlow=0
    uniformhigh=1.0
    normalloc = 0.5
    normalscale = 0.1

    cauchyloc = 1/n
    cauchyscalen = 0.004

    exponentialhigh = 15 #Symbol("b", real=True)
    exponentiallow  = 15 #Symbol("a", real=True)
    dpPrecision = 50

    beta_a = 0.1
    beta_b  = 0.1
    kumaraswamy_a = beta_a 
    kumaraswamy_b = (1.0+(beta_a-1.0)*math.pow( (beta_a+beta_b-2.0)/(beta_a-1.0), beta_a) )/beta_a 
    print(kumaraswamy_b)

    independentnormalloc1=[(float(ii)+1)/(2*n+1) for ii in range(n,0,-1)]
    independentnormalscale1=[0.05 for ii in range(n)]

    independentnormalloc2=[(float(ii)+1)/(2*n+1) for ii in range(1,n+1,1)]
    independentnormalscale2=[0.05 for ii in range(n)]


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
    d10 = D.beta.Beta(0.5,0.5)


def producedata(order,trainSize):
    global decision1,decision2
    global record_ans
    global  samplesJoint,tp_dataloader,tp_dataloader_testing,dp,decision,dp_H,decision_H
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
    elif(order=="beta"):
        print("beta_a",beta_a, "beta_b",beta_b)
        print("kumaraswamy_a",kumaraswamy_a, "kumaraswamy_b",kumaraswamy_b)
        samples1 = np.random.uniform(
            uniformlow, uniformhigh, size=(trainSize, n)
        )
        for i in range(trainSize):
            for j in range(n):
                samples1[i, j] = beta.rvs(beta_a,beta_b,  size = 1)
                while samples1[i, j] < 0 or samples1[i, j] > 1:
                    samples1[i, j] = beta.rvs(beta_a,beta_b,  size = 1)
                    
        samplesJoint = samples1
    elif(order=="arcsine"):
        print("betalow",0.5, "betahigh",0.5)
        samples1 = np.random.uniform(
            uniformlow, uniformhigh, size=(trainSize, n)
        )
        for i in range(trainSize):
            for j in range(n):
                samples1[i, j] = beta.rvs(0.5,0.5,  size = 1)
                while samples1[i, j] < 0 or samples1[i, j] > 1:
                    samples1[i, j] = beta.rvs(0.5,0.5,  size = 1)
                    
        samplesJoint = samples1
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
    plt.hist(samplesJoint,bins=500)
    plt.show()
    return samplesJoint


 
                

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
