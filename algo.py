import numpy as np
from random import sample
import multiprocessing as mp
from operator import add
from matplotlib import pyplot as plt

numbOfAgents, initialAmount, historyLength = 100, 100, 5000000
amountOfExchange = 1

poolSize = 8

def exchange(agentA, agentB, currentState):
    if currentState[agentA] >= 1:	
        currentState[agentA] -= amountOfExchange
        currentState[agentB] += amountOfExchange
    return currentState
 
def runOneHistory(_):
    currentState = [initialAmount for _ in range(numbOfAgents)]
    for _ in xrange(historyLength):
        agentA, agentB = sample(xrange(numbOfAgents), 2)
        assert agentA != agentB
        currentState = exchange(agentA, agentB, currentState) 
    return currentState

pool = mp.Pool(poolSize)
res = pool.map(runOneHistory, ['dummy'] * poolSize)
pool.close()

res = reduce(add, res)

hist, bins = np.histogram(res, bins=np.linspace(min(res), max(res), 8))
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, np.log(hist), align='center', width=width)
plt.show()

