import matplotlib # type: ignore
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from typing import List, Tuple
import numpy as np # type: ignore
from random import sample
import multiprocessing as mp
from operator import add
from functools import reduce
from itertools import starmap
from operator import add
from collections import Counter

numbOfAgents, initialAmount, historyLength = 100, 100, 10000 # type int, int, int
amountOfExchange = 1

poolSize = mp.cpu_count() - 1
sampleRepeat = 5
numberOfHistories = poolSize * sampleRepeat

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
 
def histoPlotter_entropyCalc(moneyData: List[int]) -> None:
    moneyDistribution, entropyCalculator = plotUtils(moneyData)
    histoBarCenters, histoVal, histoWidth = moneyDistribution()
    entropyValue = entropyCalculator() # to be saved into some structure for later processing
    plt.figure()
    plt.bar(histoBarCenters, histoVal, align='center', width=histoWidth, log=True, color='orange', linewidth=2) 
    plt.title('time = {:d} ; entropy = {:.3f}'.format(historyLength, entropyValue))
    plt.savefig('{:d}.png'.format(historyLength))

def runOneHistory(historyLength: int) -> List[int]:
    currentState = [initialAmount for _ in range(numbOfAgents)]
    for _ in range(historyLength):
        agentA, agentB = sample(range(numbOfAgents), 2)
        assert agentA != agentB
        currentState = exchange(agentA, agentB, currentState) 
    return currentState

def sampleAverage(historyLength: int) -> Tuple[List[int], List[int]]:
    resultsAllHistories = pool.map(runOneHistory, [historyLength] * numberOfHistories)
    assert len(resultsAllHistories) == numberOfHistories and set([len(singleHistory) for singleHistory in resultsAllHistories]) == set([numbOfAgents, ])
    individualData = list(map(lambda z: z / numberOfHistories, reduce(lambda a, b: zipWith(add, a, b), resultsAllHistories)))
    moneyData = reduce(add, resultsAllHistories)
    assert len(moneyData) == numberOfHistories * numbOfAgents
    histoPlotter_entropyCalc(moneyData)
    return individualData, moneyData
 
pool = mp.Pool(poolSize)
individualData, moneyData = sampleAverage(historyLength)
pool.close()

