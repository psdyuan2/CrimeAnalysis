import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.stats import nbinom
from tqdm import trange
# May 27
def confact(beta, gamma, n):
    return beta * (1/((2**(n-1))**gamma))
def OnePCost(h, b, d, q):
    return h*max(0, q-d) + b*max(0, d-q)
def MixtureDistribution(weightTuple, disTuple):
    return [weightTuple[0]*disTuple[0].pmf(n)+weightTuple[1]*disTuple[1].pmf(n) for n in range(1000)]
def InverseCDF(Dis, x):
    if type(Dis) == list:
        for i in range(1000):
            if np.sum(Dis[:i]) >= x:
                return i
    else:
        for i in range(1000):
            if Dis.cdf(i) >= x:
                return i

def Logl(DisObj, dataArray):
    return np.mean(np.log(DisObj.pmf(dataArray)))
def distNB(dataArray):
    return nbinom(np.sum(dataArray), len(dataArray)/(1+len(dataArray)))
def WeightPois(beta, gamma, idata, n, DisObji, DisObjj):
    return 1-(1-1/(1+np.exp(Logl(DisObjj, idata)-Logl(DisObji, idata))))*confact(beta, gamma, n)
def oOQexp1(b, h, beta, gamma, lamdJ, idata):
    alpha = WeightPois(beta, gamma, idata, len(idata), distNB(idata), poisson(lamdJ))
    mixDis = MixtureDistribution([alpha, (1-alpha)],[distNB(idata), poisson(lamdJ)])
    resQ = InverseCDF(mixDis, (b/(b+h)))
    return resQ
def costsW1(b, h, beta, gamma, lamJ, rvArray):
    costList = []
    for i in range(rvArray.shape[0]):
        data = rvArray[i,:]
        Q = oOQexp1(b, h, beta, gamma, lamJ, data[:-1])
        demand = data[-1]
        costList.append(OnePCost(h, b, demand, Q))
    return np.mean(costList)

def costW1fixed(b, h, rvArray):
    costList = []
    for i in range(rvArray.shape[0]):
        data = rvArray[i,:]
        Q = InverseCDF(distNB(data), b/(b+h))
        demand = data[-1]
        costList.append(OnePCost(h, b, demand, Q))
    return np.mean(costList)
def costStrategy1(b, h, rvArray):
    costList = []
    for i in range(rvArray.shape[0]):
        data = rvArray[i, :]
        Q = InverseCDF(distNB(data), b / (b + h))
        demand = data[-1]
        costList.append(OnePCost(h, b, demand, Q))
    return np.mean(costList)
def costStrategy2(b, h, rvArray, stateArray):
    costList = []
    for i in range(rvArray.shape[0]):
        data = rvArray[i, :-1]
        state = stateArray[:-1]
        stateZeros = [data[j] for j in range(data.shape[0]) if state[j] == 0]
        stateOnes = [data[j] for j in range(data.shape[0]) if state[j] == 1]
        # Estimate the Q
        zeroQ = InverseCDF(distNB(stateZeros), b / (b + h))
        oneQ = InverseCDF(distNB(stateOnes), b / (b + h))
        # Compute the cost
        demand = rvArray[i, -1]
        if stateArray[-1] == 0:
            costList.append(OnePCost(h, b, demand, zeroQ))
        if stateArray[-1] == 1:
            costList.append(OnePCost(h, b, demand, oneQ))
    return np.mean(costList)
def costStrategy3(b, h, beta, gamma, rvArray, stateArray):
    costList = []
    for i in range(rvArray.shape[0]):
        data = rvArray[i, :-1]
        state = stateArray[:-1]
        stateZeros = [data[j] for j in range(data.shape[0]) if state[j] == 0]
        stateOnes = [data[j] for j in range(data.shape[0]) if state[j] == 1]
        # Estimate the Q
        lamJ = np.mean(stateOnes)
        mixQ = oOQexp1(b, h, beta, gamma, lamJ, stateZeros)
        # Compute the cost
        demand = rvArray[i, -1]
        costList.append(OnePCost(h, b, demand, mixQ))
    return np.mean(costList)



def exp(periods):
    # Paras
    echo = 20
    # dataset
    # The initial period
    result = []
    # 12 periods, 12 states, 13 cols of data, the 1st col is the initial one
    stateList = np.concatenate([np.array([0]),np.random.binomial(1, 0.5, periods)])
    print(stateList)
    # generate dataset
    data = np.random.poisson(40, size=[echo,1])
    for sn in range(1, len(stateList)):
        if stateList[sn] == 0:
            data = np.concatenate([data, np.random.poisson(40, size=[echo, 1])], axis=1)
        else:
            data = np.concatenate([data, np.random.poisson(50, size=[echo, 1])], axis=1)
    for i in trange(2,data.shape[1]+1):
        rv = data[:, :i]
        tempResultRow = []
        tempResultRow.append(costW1fixed(8, 2, 40, rv))
        if stateList[i-1] == 0:
            tempResultRow.append(costsW1(8, 2, 0, 0.99, 40, rv))
        else:
            tempResultRow.append(costsW1(8, 2, 0, 0.99, 50, rv))
        if stateList[i-1] == 0:
            tempResultRow.append(costsW1(8, 2, 1, 0.99, 40, rv))
        else:
            tempResultRow.append(costsW1(8, 2, 1, 0.99, 50, rv))
        result.append(tempResultRow)
    return pd.DataFrame(result, columns=["s1","s2","s3"])
def exp2(periods):
    # Paras
    echo = 10
    # dataset
    # The initial period
    result = []
    # 12 periods, 12 states, 13 cols of data, the 1st col is the initial one
    stateList = np.concatenate([np.array([0]),np.random.binomial(1, 0.5, periods)])
    print(stateList)
    # generate dataset
    data = np.random.poisson(40, size=[echo,1])
    for sn in range(1, len(stateList)):
        if stateList[sn] == 0:
            data = np.concatenate([data, np.random.poisson(40, size=[echo, 1])], axis=1)
        else:
            data = np.concatenate([data, np.random.poisson(50, size=[echo, 1])], axis=1)



    for i in trange(2,data.shape[1]+1):
        rv = data[:, :i]
        tempResultRow = []
        tempResultRow.append(costW1fixed(8, 2, 40, rv))
        tempResultRow.append(costsW1(8, 2, 0, 0.99, 50, rv))
        tempResultRow.append(costsW1(8, 2, 1, 0.99, 50, rv))
        result.append(tempResultRow)
    return pd.DataFrame(result, columns=["s1","s2","s3"])
def optCost(lamJ, b, h):
    optQ = InverseCDF(poisson(lamJ), b/(b+h))
    cost = 0
    for x in range(200):
        cost += OnePCost(2, 8, x, optQ) * poisson(lamJ).pmf(x)

    return cost
if __name__ == "__main__":
   """a = np.random.poisson(5, [20, 13])
   for i in range(2, a.shape[1]):
       print(costsW1(8, 2, 1, 0.99, 40, a[:,:i]))"""
   pd.set_option("display.max_columns", None)
   #print(optCost(40, 8, 2))
   #print(exp(24))
   #print(WeightPois(1,0.99,a,len(a), distNB(a), poisson(40)))
   #a = np.random.poisson(50, [3,4])
   print(exp(12))
