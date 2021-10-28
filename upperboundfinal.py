import numpy as np
from scipy.stats import norm
from math import ceil

N = 5
moneyPrecision = 50
verifyIterTimes = 1000
sampleSize = 10000
welfareAsObjective = False
errorControl = True  # potential error due to money < cs

# pplTotal, pplLeft, csLeft, moneyLeft
e = np.zeros([N + 1, N + 1, moneyPrecision + 1, moneyPrecision + 1])


def cdfUniform01(x):
    return x


def cdfDoublePeak(x):
    return (
        (norm.cdf(x, loc=0.6, scale=0.1) + norm.cdf(x, loc=0.2, scale=0.1))
        - (norm.cdf(0, loc=0.6, scale=0.1) + norm.cdf(0, loc=0.2, scale=0.1))
    ) / (
        norm.cdf(1, loc=0.6, scale=0.1)
        + norm.cdf(1, loc=0.2, scale=0.1)
        - norm.cdf(0, loc=0.6, scale=0.1)
        - norm.cdf(0, loc=0.2, scale=0.1)
    )


# cdf = cdfUniform01
cdf = cdfDoublePeak


@np.vectorize
def inverseCdf(x):
    left, right = 0, 1
    while right - left > 0.001:
        mid = (left + right) / 2
        if cdf(mid) > x:
            right = mid
        else:
            left = mid
    return left


uniSamples = np.random.uniform(size=sampleSize)
samples = inverseCdf(uniSamples)
print(samples)

pf = np.zeros([moneyPrecision + 1, moneyPrecision + 1])
ps = np.zeros([moneyPrecision + 1, moneyPrecision + 1])
for o in range(moneyPrecision + 1):
    for c in range(moneyPrecision + 1):
        if c >= o:
            ps[o, c] = 1
        else:
            ps[o, c] = (1 - cdf(o / moneyPrecision)) / (1 - cdf(c / moneyPrecision))
        pf[o, c] = 1 - ps[o, c]

# welfareConditional[o] = expected welfare if o is accepted
welfareConditional = np.zeros([moneyPrecision + 1])
for o in range(moneyPrecision + 1):
    if moneyPrecision == o:
        welfareConditional[o] = 0
    else:
        res = 0
        for u in range(o + 1, moneyPrecision + 1):
            res += (u - o) / moneyPrecision * (ps[u, 0] - ps[u - 1, 0])
        welfareConditional[o] = res / (ps[moneyPrecision, 0] - ps[o, 0])

benefitDP = np.zeros([N + 1, N + 1, moneyPrecision + 1])
for pplTotal in range(1, N + 1):
    for pplLeft in range(1, pplTotal + 1):
        for moneyLeft in range(moneyPrecision + 1):
            if pplLeft == 1:
                benefitDP[pplTotal, 1, moneyLeft] = (
                    welfareConditional[moneyLeft] * ps[moneyLeft, 0]
                )
                continue
            bestRes = -1000
            for o in range(moneyLeft + 1):
                res = pf[o, 0] * benefitDP[pplTotal, pplLeft - 1, moneyLeft] + ps[
                    o, 0
                ] * (
                    benefitDP[pplTotal, pplLeft - 1, moneyLeft - o]
                    + welfareConditional[o]
                )
                if res > bestRes:
                    bestRes = res
            benefitDP[pplTotal, pplLeft, moneyLeft] = bestRes


def benefit(n):
    if welfareAsObjective:
        return benefitDP[n, n, moneyPrecision]
    else:
        return n


def getE(pplTotal):
    return e[pplTotal, pplTotal, 0, moneyPrecision]


for pplTotal in range(1, N + 1):
    print(pplTotal)
    for pplLeft in range(1, pplTotal + 1):
        for csLeft in range(moneyPrecision + 1):
            for moneyLeft in range(csLeft, moneyPrecision + 1):
                if errorControl:
                    nextMoneyPrecision = moneyPrecision - (pplTotal - 1)
                else:
                    nextMoneyPrecision = moneyPrecision
                if pplLeft == 1:
                    e[pplTotal, 1, csLeft, moneyLeft] = pf[moneyLeft, csLeft] * e[
                        pplTotal - 1,
                        pplTotal - 1,
                        moneyPrecision - moneyLeft,
                        nextMoneyPrecision,
                    ] + ps[moneyLeft, csLeft] * benefit(pplTotal)
                    continue

                bestRes = -1000
                for c in range(csLeft + 1):
                    for o in range(c, moneyLeft - csLeft + c + 1):
                        # moneyLeft-o >= csLeft-c
                        # moneyLeft-csLeft+c >= o
                        res = (
                            pf[o, c]
                            * e[
                                pplTotal - 1,
                                pplTotal - 1,
                                csLeft - c + moneyPrecision - moneyLeft,
                                nextMoneyPrecision,
                            ]
                            + ps[o, c]
                            * e[pplTotal, pplLeft - 1, csLeft - c, moneyLeft - o]
                        )
                        if res > bestRes:
                            bestRes = res
                e[pplTotal, pplLeft, csLeft, moneyLeft] = bestRes


def calcBenefit(bids):
    coalitionSize = len(bids)
    offers = [1 / coalitionSize] * coalitionSize
    remainingBids = []
    for i in range(coalitionSize):
        if bids[i] > offers[i]:
            remainingBids.append(bids[i])
    remainingSize = len(remainingBids)
    if remainingSize == coalitionSize:
        return benefit(coalitionSize)
    elif remainingSize <= 1:
        return 0
    else:
        return calcBenefit(remainingBids)


def verify(n):
    benefitSum = 0
    for iterIndex in range(verifyIterTimes):
        bids = np.random.choice(samples, size=n)
        benefitSum += calcBenefit(bids)
    print(benefitSum / verifyIterTimes)


for report in range(2, N + 1):
    res = getE(report)
    print(res, report - res)
    verify(report)
    print()
