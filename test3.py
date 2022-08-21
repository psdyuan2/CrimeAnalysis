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
def InverseCDF2(Dis, weights, x):
    for i in range(1000):
        if np.sum(weights[0] * Dis[0].cdf(i) + weights[1] * Dis[1].cdf(i)) >= x:
            print(i)
            return i


def Logl(DisObj, dataArray):
    return np.mean(np.log(DisObj.pmf(dataArray)))
def distNB(dataArray):
    return nbinom(np.sum(dataArray), len(dataArray)/(1+len(dataArray)))
def WeightPois(beta, gamma, idata, n, DisObji, DisObjj):
    return 1-(1-1/(1+np.exp(Logl(DisObjj, idata)-Logl(DisObji, idata))))*confact(beta, gamma, n)
def oOQexp1(b, h, beta, gamma, jdata, idata):
    alpha = WeightPois(beta, gamma, idata, len(idata), distNB(idata), distNB(jdata))
    mixQ = InverseCDF2(Dis=[distNB(idata), distNB(jdata)],weights=[alpha, (1-alpha)],x=b / (b + h))
    return mixQ, alpha

def costStrategy1(b, h, rvArray, stateArray):
    costList = []
    data = rvArray[:-1]
    state = stateArray[:-1]
    stateZeros = [data[j] for j in range(data.shape[0]) if state[j] == 0]
    Q = InverseCDF(distNB(stateZeros), b / (b + h))
    demand = data[-1]
    costList.append(OnePCost(h, b, demand, Q))
    return np.mean(costList)
def costStrategy2(b, h, rvArray, stateArray):
    costList = []
    data = rvArray[:-1]
    state = stateArray[:-1]
    stateZeros = [data[j] for j in range(data.shape[0]) if state[j] == 0]
    stateOnes = [data[j] for j in range(data.shape[0]) if state[j] == 1]
    # Estimate the Q
    zeroQ = InverseCDF(distNB(stateZeros), b / (b + h))
    oneQ = InverseCDF(distNB(stateOnes), b / (b + h))
    # Compute the cost
    demand = rvArray[-1]
    if stateArray[-1] == 0:
        costList.append(OnePCost(h, b, demand, zeroQ))
    if stateArray[-1] == 1:
        costList.append(OnePCost(h, b, demand, oneQ))
    return np.mean(costList)
def costStrategy3(b, h, beta, gamma, rvArray, stateArray):
    costList = []
    alpha_list = []
    data = rvArray[:-1]
    state = stateArray[:-1]
    stateZeros = [data[j] for j in range(data.shape[0]) if state[j] == 0]
    stateOnes = [data[j] for j in range(data.shape[0]) if state[j] == 1]
    # Estimate the Q
    zeroQ = InverseCDF(distNB(stateZeros), b / (b + h))
    mixQ, alpha = oOQexp1(b, h, beta, gamma, stateZeros, stateOnes)
    # Compute the cost
    demand = rvArray[-1]
    if stateArray[-1] == 0:
        costList.append(OnePCost(h, b, demand, zeroQ))
    if stateArray[-1] == 1:
        costList.append(OnePCost(h, b, demand, mixQ))
    alpha_list.append(alpha)
    averageAlpha = np.mean(alpha_list)
    return np.mean(costList), averageAlpha
def optCost(lamJ, b, h):
    optQ = InverseCDF(poisson(lamJ), b/(b+h))
    cost = 0
    for x in range(200):
        cost += OnePCost(h, b, x, optQ) * poisson(lamJ).pmf(x)
    return cost



def exp2(periods):
    # Paras
    echo = 20
    # dataset
    # The initial period
    result = []
    # 12 periods, 12 states, 13 cols of data, the 1st col is the initial one
    stateList = np.concatenate([np.array([0,1]),np.random.binomial(1, 0.5, periods)])
    # generate dataset
    data = np.random.poisson(40, size=[echo,1])
    data = np.concatenate([data, np.random.poisson(50, size=[echo,1])], axis=1)
    for sn in range(2, len(stateList)):
        if stateList[sn] == 0:
            data = np.concatenate([data, np.random.poisson(40, size=[echo, 1])], axis=1)
        else:
            data = np.concatenate([data, np.random.poisson(50, size=[echo, 1])], axis=1)
    for i in trange(3,data.shape[1]+1):
        rv = data[:, :i]
        state = stateList[:i]
        tempResultRow = []
        tempResultRow.append(costStrategy1(8, 2, rv, state))
        tempResultRow.append(costStrategy2(8,2,rv,state))
        cost3, averageAlpha = costStrategy3(8,2,1,0.66,rv,state)
        tempResultRow.append(cost3)
        tempResultRow.append(averageAlpha)
        result.append(tempResultRow)
    return pd.DataFrame(result, columns=["s1","s2","s3","Average alpha"])
def exp3(ms):
    from AreaObject import Area
    # Hyper-parameters
    LockdownState=[0,1,1,1,0,0,0,0,1,1,0,1,1,1,0,0,0,0,0]
    data=Area(msoa=ms).GetFrequency()[17:]
    result=[]
    for i in trange(3,data.shape[0]+1):
        rv=data[:i]
        state=LockdownState[:i]
        temResultRow=[]
        temResultRow.append(costStrategy1(8,2,rv,state))
        temResultRow.append(costStrategy2(8,2,rv,state))
        cost3,averageAlpha=costStrategy3(8,2,1,0.66,rv,state)
        temResultRow.append(cost3)
        temResultRow.append(averageAlpha)
        result.append(temResultRow)
    tempDF=pd.DataFrame(result,columns=['s1','regret1','s2','regret2','s3','regret3','Average alpha'])
    resDF=tempDF.mean(axis=0)
    return resDF
def exp3part2():
    mList = pd.read_csv('M_list.csv')
    for i in range(len(m))
if __name__ == "__main__":
   print(exp3())
