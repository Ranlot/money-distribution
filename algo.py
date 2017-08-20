import matplotlib # type: ignore
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from typing import List, Tuple, Iterator
from typing import NamedTuple # use this instead of collections to avoid Any type
import numpy as np # type: ignore
from random import sample
import multiprocessing as mp
from operator import add
from functools import reduce
from itertools import starmap
from operator import add
from collections import Counter
import datetime

HistogramData = Tuple[List[float], List[float], float]
DataForLength = NamedTuple('DataForLength', [('historyLength', int), ('moneyData', List[int]), ('individualData', List[float])])

poolSize, sampleRepeat  = mp.cpu_count() - 1, 40
numberOfHistories = poolSize * sampleRepeat

def timeMe(functionToTime):
    def timerClosure(*args, **kw):
        timeStart = datetime.datetime.now()
        result = functionToTime(*args, **kw)
        timeEnd = datetime.datetime.now()
        print('{:d} {}'.format(args[0], timeEnd - timeStart))
        return result
    return timerClosure

def zipWith(f, *coll):
    return starmap(f, zip(*coll))

def exchange(agentA: int, agentB: int, currentState: List[int]) -> List[int]:
    if currentState[agentA] >= 1:	
        currentState[agentA] -= amountOfExchange
        currentState[agentB] += amountOfExchange
    return currentState

def wealthDistributionPlotter(result: DataForLength) -> None:
    def moneyDistribution():
        hist, bins = np.histogram(result.moneyData, bins=np.linspace(min(result.moneyData), max(result.moneyData), 8))
        width = 0.9 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        return center, hist, width 
    histoBarCenters, histoVal, histoWidth = moneyDistribution() # type: HistogramData
    plt.figure()
    plt.bar(histoBarCenters, histoVal, align='center', width=histoWidth, log=True, color='orange', linewidth=1, edgecolor='k') 
    plt.title('time = {:d} ; entropy = {:.2f}'.format(result.historyLength, entropyCalculator(result.moneyData)))
    plt.ylim([0.1 * 10**1, 1 * 10**6]); plt.xlim([0, 1500])
    plt.savefig('wealthDistribution.{:d}.png'.format(result.historyLength)); plt.close()

def individualPlotter(result: DataForLength, initialAmount: int) -> None:
    relativeDiffToInitial = [100 * (initialAmount - x) / initialAmount for x in result.individualData]
    plt.figure(); plt.subplot(211)
    plt.bar(range(numbOfAgents), relativeDiffToInitial)
    plt.subplot(212); plt.bar(range(numbOfAgents), sorted(map(abs, relativeDiffToInitial), reverse=True))
    plt.suptitle('Average wealth of agents; time = {:d}\nRelative difference to initial amount in %'.format(result.historyLength))
    plt.savefig('individual.{:d}.png'.format(result.historyLength)); plt.close()
 
def entropyPlotter(allResults: List[DataForLength]) -> None:
    entropyData = [(result.historyLength, entropyCalculator(result.moneyData)) for result in allResults]
    plt.figure()
    plt.scatter(*zip(*entropyData), marker='*', s=200, c='g'); plt.xscale('log'); plt.xlabel('time')
    plt.title('Entropy evolution in time'); plt.savefig('entropy.png'); plt.close()

def entropyCalculator(moneyData: List[int]) -> float:
    probabilities = [x / len(moneyData) for x in list(Counter(moneyData).values())]
    np.testing.assert_array_almost_equal(sum(probabilities), 1, 10)
    return -sum([p * np.log(p) for p in probabilities])

def runOneHistory(historyLength: int) -> List[int]:
    currentState = [initialAmount for _ in range(numbOfAgents)]
    for _ in range(historyLength):
        agentA, agentB = sample(range(numbOfAgents), 2)
        assert agentA != agentB
        currentState = exchange(agentA, agentB, currentState) 
    return currentState

def sanityChecks(resultsAllHistories, moneyData: List[int]) -> bool:
    check1 = len(resultsAllHistories) == numberOfHistories and set([len(singleHistory) for singleHistory in resultsAllHistories]) == set([numbOfAgents, ]) # type: bool
    check2 = len(moneyData) == numberOfHistories * numbOfAgents # type: bool
    return check1 and check2

@timeMe
def sampleAverage(historyLength: int) -> DataForLength:
    pool = mp.Pool(poolSize)
    resultsAllHistories = pool.map(runOneHistory, [historyLength] * numberOfHistories)
    pool.close()
    individualData = list(map(lambda z: z / numberOfHistories, reduce(lambda a, b: zipWith(add, a, b), resultsAllHistories))) # type: List[float]
    moneyData = reduce(add, resultsAllHistories) # type: List[int]
    # individualData contains the average wealth held for all agents averaged over all the histories
    # moneyData simply accumulates all the lists into a single long list of wealth regardless of the precise agent
    assert sanityChecks(resultsAllHistories, moneyData)    
    return DataForLength(historyLength=historyLength, moneyData=moneyData, individualData=individualData)

if __name__ == "__main__":
    numbOfAgents, initialAmount, amountOfExchange = 100, 100, 1
    historyLengthsToRun = map(int, np.logspace(np.log10(1), np.log10(20000001), 22))
    #historyLengthsToRun = [100, 1000, 2000]
    allResults = list(map(sampleAverage, historyLengthsToRun)) # type: List[DataForLength]
    # plotting ---------
    entropyPlotter(allResults)
    for result in allResults:
        wealthDistributionPlotter(result)
        individualPlotter(result, initialAmount)
    # plotting ---------
    print("Done!!!")

