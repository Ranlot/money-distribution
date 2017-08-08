import matplotlib # type: ignore
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from typing import List, Tuple, Callable, Iterator
import numpy as np # type: ignore
from random import sample
import multiprocessing as mp
from operator import add
from functools import reduce
from itertools import starmap
from operator import add
from collections import Counter

HistogramData = Tuple[List[float], List[float], float]

poolSize, sampleRepeat  = mp.cpu_count() - 1, 5
numberOfHistories = poolSize * sampleRepeat

def zipWith(f, *coll):
    return starmap(f, zip(*coll))

def exchange(agentA: int, agentB: int, currentState: List[int]) -> List[int]:
    if currentState[agentA] >= 1:	
        currentState[agentA] -= amountOfExchange
        currentState[agentB] += amountOfExchange
    return currentState

def plotUtils(moneyData: List[int]) -> Tuple[Callable[[], HistogramData], Callable[[], float]]:
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
 
'''
def histoPlotter_entropyCalc(moneyData: List[int], historyLength: int) -> float:
    moneyDistribution, entropyCalculator = plotUtils(moneyData) # type: Callable[[], HistogramData], Callable[[], float]
    histoBarCenters, histoVal, histoWidth = moneyDistribution() # type: HistogramData
    entropyValue = entropyCalculator() # type: float
    plt.figure()
    plt.bar(histoBarCenters, histoVal, align='center', width=histoWidth, log=True, color='orange', linewidth=2) 
    plt.title('time = {:d} ; entropy = {:.3f}'.format(historyLength, entropyValue))
    plt.savefig('{:d}.png'.format(historyLength))
    return entropyValue
'''

def runOneHistory(historyLength: int) -> List[int]:
    currentState = [initialAmount for _ in range(numbOfAgents)]
    for _ in range(historyLength):
        agentA, agentB = sample(range(numbOfAgents), 2)
        assert agentA != agentB
        currentState = exchange(agentA, agentB, currentState) 
    return currentState

# TODO: understand how to annotate resultsAllHistories; dependent types to replace the runtime asserts
def sanityChecks(resultsAllHistories, moneyData: List[int]) -> bool:
    check1 = len(resultsAllHistories) == numberOfHistories and set([len(singleHistory) for singleHistory in resultsAllHistories]) == set([numbOfAgents, ]) # type: bool
    check2 = len(moneyData) == numberOfHistories * numbOfAgents # type: bool
    return check1 and check2

def sampleAverage(historyLength: int) -> Tuple[int, List[int], List[float]]:
    print(historyLength)
    pool = mp.Pool(poolSize)
    resultsAllHistories = pool.map(runOneHistory, [historyLength] * numberOfHistories)
    pool.close()
    individualData = list(map(lambda z: z / numberOfHistories, reduce(lambda a, b: zipWith(add, a, b), resultsAllHistories)))
    moneyData = reduce(add, resultsAllHistories)
    # individualData contains the average wealth held for all agents averaged over all the histories
    # moneyData simply accumulates all the lists into a single long list of wealth regardless of the precise agent
    assert sanityChecks(resultsAllHistories, moneyData)    
    return historyLength, moneyData, individualData

if __name__ == "__main__":
    numbOfAgents, initialAmount, amountOfExchange = 100, 100, 1
    historyLengthsToRun = [100, 1000, 2000]
    res = map(sampleAverage, historyLengthsToRun) # type: Iterator[Tuple[int, List[int], List[float]]]
    tt = list(res)

