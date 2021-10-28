import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as D
from torch.utils.data import DataLoader, TensorDataset

n = 5
epochs = 100
supervisionEpochs = 5
lr = 0.001
log_interval = 20
trainSize = 10000
penaltyLambda = 10
doublePeakHighMean = 0.9
doublePeakLowMean = 0.1
doublePeakStd = 0.2

d1 = D.normal.Normal(doublePeakLowMean, doublePeakStd)
d2 = D.normal.Normal(doublePeakHighMean, doublePeakStd)
distributionRatio = (d1.cdf(1) + d2.cdf(1) - d1.cdf(0) - d2.cdf(0)) / 2
distributionBase = d1.cdf(0) + d2.cdf(0)


def cdf(x, i=None):
    return (d1.cdf(x) + d2.cdf(x) - distributionBase) / 2 / distributionRatio


# def cdf(x, i=None):
#     if x < 0.1:
#         return 0
#     if x <= 0.2:
#         return 0.5 * (x - 0.1) / 0.1
#     if x < 0.8:
#         return 0.5
#     if x < 0.9:
#         return 0.5 + 0.5 * (x - 0.8) / 0.1
#     return 1


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
tp_dataset = TensorDataset(tp_tensor[: round(trainSize * 0.9)])
tp_dataset_testing = TensorDataset(tp_tensor[round(trainSize * 0.9) :])
tp_dataloader = DataLoader(tp_dataset, batch_size=64, shuffle=True)
tp_dataloader_testing = DataLoader(tp_dataset_testing, batch_size=128, shuffle=False)


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

# optimizer = optim.SGD(model.parameters(), lr=lr)
optimizer = optim.Adam(model.parameters(), lr=lr)


def bitsToPayments(bits):
    if sum(bits).item() == 0:
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
    newBits = tp >= payments
    if torch.equal(newBits, bits):
        return bits
    else:
        return tpToBits(tp, newBits)


def tpToPayments(tp):
    return bitsToPayments(tpToBits(tp))


def tpToTotalDelay(tp):
    return n - sum(tpToBits(tp).type(torch.float32))


dpPrecision = 100
# howManyPpl left, money left, yes already
dp = np.zeros([n + 1, dpPrecision + 1, n + 1])
decision = np.zeros([n + 1, dpPrecision + 1, n + 1], dtype=np.uint8)
# ppl = 0 left
for yes in range(n + 1):
    for money in range(dpPrecision + 1):
        if money == 0:
            dp[0, 0, yes] = 0
        else:
            dp[0, money, yes] = yes
for ppl in range(1, n + 1):
    for yes in range(n + 1):
        for money in range(dpPrecision + 1):
            minSoFar = 1_000_000
            for offerIndex in range(money + 1):
                offer = offerIndex / dpPrecision
                res = (1 - cdf(offer)) * dp[
                    ppl - 1, money - offerIndex, min(yes + 1, n)
                ] + cdf(offer) * (1 + dp[ppl - 1, money, yes])
                if minSoFar > res:
                    minSoFar = res
                    decision[ppl, money, yes] = offerIndex
            dp[ppl, money, yes] = minSoFar


def dpSupervisionRule(tp):
    tp = list(tp.numpy())
    bits = [0] * n
    payments = [0] * n
    money = dpPrecision
    yes = 0
    for i in range(n):
        offerIndex = decision[n - i, money, yes]
        offer = offerIndex / dpPrecision
        if tp[i] > offer:
            money -= offerIndex
            yes += 1
            bits[i] = 1
            payments[i] = offer
        else:
            bits[i] = 0
            payments[i] = 1
    if money > 0:
        bits = [0] * n
        payments = [1] * n
    bits = torch.tensor(bits, dtype=torch.uint8)
    payments = torch.tensor(payments, dtype=torch.float32)
    # print()
    # print(tp)
    # print(bits)
    # print(payments)
    # print()
    return (bits, payments)


def dpDelay(tp):
    bits, payments = dpSupervisionRule(tp)
    totalPayment = torch.dot(bits.type(torch.float32), payments).item()
    if totalPayment < 0.98:
        return n
    else:
        return n - sum(bits).item()


def costSharingSupervisionRule(tp):
    tp = list(tp.numpy())
    for k in range(n, -1, -1):
        if k == 0:
            break
        bits = [1 if tp[i] >= 1 / k else 0 for i in range(n)]
        if sum(bits) == k:
            break
    if k == 0:
        payments = [1] * n
    else:
        payments = [1 / k if bits[i] == 1 else 1 for i in range(n)]
    bits = torch.tensor(bits, dtype=torch.uint8)
    payments = torch.tensor(payments, dtype=torch.float32)
    return (bits, payments)


def costSharingDelay(tp):
    return n - sum(costSharingSupervisionRule(tp)[0]).item()


allBits = [torch.tensor(bits) for bits in itertools.product([0, 1], repeat=n)]

runningLossNN = []
runningLossCS = []
runningLossDP = []


def recordAndReport(name, source, loss, n=100):
    source.append(loss)
    realLength = min(n, len(source))
    avgLoss = sum(source[-n:]) / realLength
    print(f"{name} ({realLength}): {avgLoss}")


def supervisionTrain(epoch, supervisionRule):
    model.train()
    for batch_idx, (tp_batch,) in enumerate(tp_dataloader):
        optimizer.zero_grad()
        loss = 0
        for tp in tp_batch:
            bits, payments = supervisionRule(tp)
            # print()
            # print("supervision")
            # print(tp)
            # print(bits)
            # print()
            # print(payments)
            # print(bitsToPayments(bits))
            # print()
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


def train(epoch):
    model.train()
    for batch_idx, (tp_batch,) in enumerate(tp_dataloader):
        optimizer.zero_grad()
        penalty = 0
        for bitsMoreOnes in allBits:
            for i in range(n):
                if bitsMoreOnes[i] == 1:
                    bitsLessOnes = bitsMoreOnes.clone()
                    bitsLessOnes[i] = 0
                    penalty = penalty + sum(
                        torch.relu(
                            bitsToPayments(bitsMoreOnes) - bitsToPayments(bitsLessOnes)
                        )
                    )
        loss = penalty * penaltyLambda
        # costSharingLoss = 0
        # dpLoss = 0
        for tp in tp_batch:
            # costSharingLoss += costSharingDelay(tp)
            # dpLoss += dpDelay(tp)
            # print()
            # print("---")
            # print(tp)
            # print(costSharingSupervisionRule(tp))
            # print(dpSupervisionRule(tp))
            # print(costSharingDelay(tp), dpDelay(tp))
            # print()
            for i in range(n):
                tp1 = tp.clone()
                tp1[i] = 1
                tp0 = tp.clone()
                tp0[i] = 0
                offer = tpToPayments(tp1)[i]
                delay1 = tpToTotalDelay(tp1)
                delay0 = tpToTotalDelay(tp0)
                loss = loss + (1 - cdf(offer)) * delay1 + cdf(offer) * delay0

        loss = loss / len(tp_batch) / n
        # costSharingLoss /= len(tp_batch)
        # dpLoss /= len(tp_batch)
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
            # recordAndReport("NN", runningLossNN, loss.item())
            # recordAndReport("CS", runningLossCS, costSharingLoss)
            # recordAndReport("DP", runningLossDP, dpLoss)
            # print(dp[n, dpPrecision, 0])
            # print(penalty.item())
            # print(distributionRatio)
            # for i in range(n, 0, -1):
            #     print(
            #         tpToPayments(
            #             torch.tensor([1] * i + [0] * (n - i), dtype=torch.float32)
            #         )
            #     )


def test():
    model.eval()
    with torch.no_grad():
        for batch_idx, (tp_batch,) in enumerate(tp_dataloader_testing):
            costSharingLoss = 0
            dpLoss = 0
            nnLoss = 0
            for tp in tp_batch:
                costSharingLoss += costSharingDelay(tp)
                dpLoss += dpDelay(tp)
                nnLoss += tpToTotalDelay(tp)
            costSharingLoss /= len(tp_batch)
            dpLoss /= len(tp_batch)
            nnLoss /= len(tp_batch)
            if batch_idx % log_interval == 0:
                recordAndReport("NN", runningLossNN, nnLoss)
                recordAndReport("CS", runningLossCS, costSharingLoss)
                recordAndReport("DP", runningLossDP, dpLoss)
                print(dp[n, dpPrecision, 0])
                for i in range(n, 0, -1):
                    print(
                        tpToPayments(
                            torch.tensor([1] * i + [0] * (n - i), dtype=torch.float32)
                        )
                    )


for epoch in range(1, supervisionEpochs + 1):
    print(distributionRatio)
    # supervisionTrain(epoch, costSharingSupervisionRule)
    supervisionTrain(epoch, dpSupervisionRule)
for epoch in range(1, epochs + 1):
    train(epoch)
    test()
