import numpy as np
from random import sample
import multiprocessing as mp
from operator import add
from matplotlib import pyplot as plt
from functools import reduce
from itertools import starmap
from operator import add
from collections import Counter

numbOfAgents, initialAmount, historyLength = 100, 100, 50000
amountOfExchange = 1

poolSize = mp.cpu_count() - 1
sampleRepeat = 2 # number of independent history to run in units of the poolSize

def zipWith(f, *coll):
    return starmap(f, zip(*coll))

def exchange(agentA, agentB, currentState):
    if currentState[agentA] >= 1:	
        currentState[agentA] -= amountOfExchange
        currentState[agentB] += amountOfExchange
    return currentState

def plotUtils(moneyData):
    def moneyDistribution():
        hist, bins = np.histogram(moneyData, bins=np.linspace(min(moneyData), max(moneyData), 8))
        width = 0.85 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        return center, hist, width
    def entropyCalculator():
        probabilities = [x / len(moneyData) for x in list(Counter(moneyData).values())]
        np.testing.assert_array_almost_equal(sum(probabilities), 1, 10)
        return -sum([p * np.log(p) for p in probabilities])
    return moneyDistribution, entropyCalculator
 
def histoPlotter_entropyCalc(moneyData):
    moneyDistribution, entropyCalculator = plotUtils(moneyData)
    histoBarCenters, histoVal, histoWidth = moneyDistribution()
    entropyValue = entropyCalculator() # to be saved into some structure for later processing
    plt.figure()
    plt.bar(histoBarCenters, histoVal, align='center', width=histoWidth, log=True, color='orange', linewidth=2) 
    plt.title('time = {:d} ; entropy = {:.3f}'.format(historyLength, entropyValue))
    plt.savefig('{:d}.png'.format(historyLength))

def runOneHistory(historyLength):
    currentState = [initialAmount for _ in range(numbOfAgents)]
    for _ in range(historyLength):
        agentA, agentB = sample(range(numbOfAgents), 2)
        assert agentA != agentB
        currentState = exchange(agentA, agentB, currentState) 
    return currentState

def sampleAverage(historyLength):
    moneyData = reduce(add, pool.map(runOneHistory, [historyLength] * sampleRepeat * poolSize))
    assert len(moneyData) == sampleRepeat * poolSize * numbOfAgents
    histoPlotter_entropyCalc(moneyData)
    return moneyData
    
pool = mp.Pool(poolSize)
moneyData = sampleAverage(historyLength)
pool.close()

# for testing 
# moneyData = map(runOneHistory, [historyLength] * 5)
# list(reduce(lambda a, b: zipWith(add, a, b), moneyData))[:10]

