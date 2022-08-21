"""Basic statistical functions"""
import numpy as np
from scipy.stats import poisson
from src.AreaObject import Area




def EmpiricalPDF(x, sample, m):
    n=sample.size
    f, b=np.histogram(sample, bins=m)
    f=f/n
    for k in range(m-1):
        f[k+1]+=f[k]
    y=np.zeros(x.size)
    for i in range(1, m):
        d=np.where((x>b[i-1])&(x<=b[i]))
        y[d]=f[i-1]
    d=np.where(x>b[m-1])
    y[d]=1
    return y
# This likelihood func is only for Poisson dis
def LogLikelihood(disType,para,data):
    if disType == "Poisson":
        return np.mean(np.log([poisson(np.mean(para)).cdf(x) for x  in data]))
# This likelihood func is for general dis
def Logl(DisObj, dataArray):
    np.mean(np.log(DisObj.cdf(dataArray)))
def LogLikelihoodTotal(disType,paraList,data):
    if disType == "Poisson":
        return np.sum(np.log([poisson(np.mean(paraList[0])).cdf(x) for x  in data]))
def LogLikeLihoodEmp(disData,testData):
    return np.mean(np.log([EmpiricalPDF(x, disData, max(disData)) for x in testData]))

"""
Data selection functions
"""
def PickData(dataPairs, criterion):
    resultDataPairs = []
    for pair in dataPairs:
        if pair[1] == criterion:
            resultDataPairs.append(pair)
    return np.array(resultDataPairs)

def MakeDataSets(n, dataPairs):
    tempData = dataPairs[n,:]
    return PickData(tempData, 0), PickData(tempData, 1)

"""
Basic statistical functions
"""
def confact(beta, gamma, n):
    return beta * (1/((2**(n-1))**gamma))
def PrimaryWeightPois(beta, gamma, idata, jdata):
    tempN = len(idata)
    lam_i = np.mean(idata)
    lam_j = np.mean(jdata)
    return (
        1 - (
        1 - (
        1/(
        1+np.exp(LogLikelihood('Poisson', lam_j, idata) - LogLikelihood('Poisson', lam_i, idata))
    )
    )
    ) * confact(beta, gamma, tempN)
    )

def InverseCDF(Dis, x):
    if type(Dis) == list:
        for i in range(1000):
            if np.sum(Dis[:i]) >= x:
                return i
    else:
        for i in range(1000):
            if Dis(i) >= x:
                return i



def estPoisQ(critfrac, beta, gamma, idata, jdata):
    alpha = PrimaryWeightPois(beta, gamma, idata, jdata)
    lam_i = np.mean(idata)
    lam_j = np.mean(jdata)
    n = len(idata)
    if (jdata != []) and (idata != []):
        return InverseCDF(poisson(alpha*np.mean(idata)+(1-alpha)*np.mean(jdata)).cdf, critfrac)

def RandomstatePoisQ(critfrac, beta, gamma, statecrime):
    """
    Input
    :param critfrac:
    :param beta:
    :param gamma:
    :param statecrime: A table containing data pairs. [stete of lockdown or unlockdown, the crime cases]
    :return:The quantity
    """
    curstate = 0
    idata = PickData(statecrime, curstate)
    curstate = 1
    jdata = PickData(statecrime, curstate)
    return estPoisQ(critfrac, beta, gamma, idata, jdata)

def estPoisQMixture1(critfrac, beta, gamma, idata, jdata):
    """
    This method uses the binominal random to generate a binary sequence. If
    the value in the sequence is 1, it will randomly fetch one data from idata;
    Else, it will randomly fetch a sample from j data.
    :param critfrac:
    :param beta:
    :param gamma:
    :param idata:
    :param jdata:
    :return:
    """
    alpha = PrimaryWeightPois(beta, gamma, idata, jdata)
    lam_i = np.mean(idata)
    lam_j = np.mean(jdata)
    n = len(idata)
    mixedDis = MixtureDistribution([alpha, 1-alpha], [poisson(lam_i), poisson(lam_j)], 1000)
    return InverseCDF(mixedDis, critfrac)

def StraightQ(critfrac, idata):
    dis = poisson(np.mean(idata)).cdf
    return InverseCDF(dis, critfrac)
"""
Cost functions
"""
def OnePCost(h, b, d, q):
    return h*max(0, q-d) + b*max(0, d-q)
def CostOverEpochs(b, h, Crimes, qs):
    resList = []
    for i in range(len(Crimes)):
        resList.append(OnePCost(h, b, Crimes[i], qs[i]))
    return resList

def GenSet(statecrime, n, state):
    return PickData(statecrime[:n, :], state)

def mixtureCost(b, h, beta, gamma, stateCrime):
    totalCost = 0
    tempQ = RandomstatePoisQ(b/(b+h), beta, gamma, stateCrime)
    for i in range(len(stateCrime)):
        totalCost += OnePCost(h, b, stateCrime[i,1], tempQ)
    return totalCost
def StraightCost(b, h, stateCrime):
    totalCost = 0
    for i in range(len(stateCrime)):
        tempQ = StraightQ(b/(b+h), stateCrime[:i, 1])
        totalCost += OnePCost(h, b, stateCrime[i, 1], tempQ)
    return totalCost
def TwoDisCostByPeriod(b, h, stateCrime):
    """
        I used i and i+1 instead of 'i-1 and i'
    """
    resList = []
    for i in range(len(stateCrime)):
        tempStraightQ = StraightQ(b/(b+h), GenSet(stateCrime, i, i+1))
        tempCost = OnePCost(h, b, stateCrime[i,1], tempStraightQ)
        resList.append(tempCost)
    return np.array(resList)
# May 27
def MixtureDistribution(weightTuple, disTuple, r):
    return [weightTuple[0]*disTuple[0].pmf(n)+weightTuple[1]*disTuple[1].pmf(n) for n in range(1000)]
def InverseCDF(Dis, x):
    if type(Dis) == list:
        for i in range(1000):
            if np.sum(Dis[:i]) >= x:
                return i
    else:
        for i in range(1000):
            if Dis(i) >= x:
                return i
from scipy.stats import nbinom
def Logl(DisObj, dataArray):
    return np.mean(np.log(DisObj.cdf(dataArray)))
def distNB(dataArray):
    return nbinom(np.sum(dataArray), len(dataArray)/(1+len(dataArray)))
def WeightPois(beta, gamma, idata, n, DisObji, DisObjj):
    return (1-(1-1/(1+np.exp(Logl(DisObjj, idata)-Logl(DisObji, idata))))*confact(beta, gamma, n))
def oOQexp1(b, h, beta, gamma, lamdJ, idata):
    alpha = WeightPois(beta, gamma, idata, len(idata), distNB(idata), poisson(lamdJ))
    mixDis = MixtureDistribution([alpha, (1-alpha)],[distNB(idata), poisson(lamdJ)])
    resQ = InverseCDF(mixDis, (b/(b+h)))
    return resQ
def costsW1(b, h, beta, gamma, lamJ, rv):
    costList = []
    for i in range(1, len(rv)+1):
        data = rv[:-i]
        Q = oOQexp1(b, h, beta, gamma, lamJ, data)
        demands = data[-1]
        costList.append(OnePCost(h, b, demands, Q))
    return np.mean(costList)
def costW1fixed(b, h, lamJ, rv):
    costList = []
    Q = InverseCDF(poisson(lamJ), b/(b+h))
    demand = rv[-1]
    for i in range(len(rv)):
        costList.append(OnePCost(h, b, demand, Q))
    return np.mean(costList)
def exp():
    # dataset
    result = []
    data = np.random.poisson(40, size=[100, 13])
    for i in range(1,len(data.shape[13])):
        rv = data[:, i]
        tempResultRow = []
        tempResultRow.append(costW1fixed(8, 2, 50, rv))
        tempResultRow.append(costsW1(8, 2, 0, 0.99, 50, rv))
        tempResultRow.append(costsW1(8,2,0,0.99,rv))
        result.append(tempResultRow)
    return result








if __name__ == "__main__":
    a = np.array([50, 50, 5, 7, 23, 55])
    for i in range(100):
        print((1, 0.99, i))