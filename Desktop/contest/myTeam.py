# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game



#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='QLearningAgent', second='QLearningAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.
  
    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class DummyAgent(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at baselineTeam.py for more details about how to
    create an agent as this is the bare minimum.
    """

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).
    
        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)
    
        IMPORTANT: This method may run for at most 15 seconds.
        """

        '''
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.registerInitialState in captureAgents.py.
        '''
        CaptureAgent.registerInitialState(self, gameState)

        '''
        Your initialization code goes here, if you need any.
        '''

    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        actions = gameState.getLegalActions(self.index)

        '''
        You should change this in your own agent.
        '''

        return random.choice(actions)

class QLearningAgent(CaptureAgent):

    def __init__(self, index, epsilon = 0.1, gamma = 0.9, alpha = 0.8):
        self.index = index
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.brain = util.Counter()
        self.observationHistory = []


    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)


    def getQValue(self, gameState, action):
        return self.brain[(gameState, action)]
        '''
        nextState = gameState.generateSuccessor(self.index, action)
        pos = nextState.getAgentPosition(self.index)
        foods = self.getFood(gameState)
        furthestFood = self.getFurthestFood(gameState)
        if pos in foods:
            return 100
        return float(1) / self.getMazeDistance(pos, furthestFood)
        '''

    def getBestAction(self, gameState):
        bestAction = None
        maxVal = 0
        legalActions = gameState.getLegalActions(self.index)
        legalActions.remove('Stop')
        for action in legalActions:
            qValTmp = self.getQValue(gameState, action)
            if qValTmp > maxVal:
                bestAction = action
                maxVal = qValTmp
        if bestAction == None:
            return random.choice(legalActions)
        else:
            return bestAction

    def getFurthestFood(self, gameState):
        furthestFood = None
        foods = self.getFood(gameState).asList()
        curPos = gameState.getAgentPosition(self.index)
        if len(foods) == 0:
            return (0, 0)
        maxDis = 0
        for food in foods:
            if self.getMazeDistance(curPos, food) > maxDis:
                maxDis = self.getMazeDistance(curPos, food)
                furthestFood = food
        return furthestFood

    def update(self, gameState, action, nextState, reward):
        firstPart = self.getQValue(gameState, action)
        if len(nextState.getLegalActions(self.index)) == 0:
            tmp = reward - firstPart
        else:
            tmp = reward + (self.gamma * max([self.getQValue(nextState, nextAction) for nextAction in nextState.getLegalActions(self.index)])) - firstPart
        secondPart = self.alpha * tmp
        self.brain[(gameState, action)] = firstPart + secondPart

    def getFurthestFood(self, gameState):
        furthestFood = None
        foods = self.getFood(gameState).asList()
        curPos = gameState.getAgentPosition(self.index)
        if len(foods) == 0:
            return (0, 0)
        maxDis = 0
        for food in foods:
            if self.getMazeDistance(curPos, food) > maxDis:
                maxDis = self.getMazeDistance(curPos, food)
                furthestFood = food
        return furthestFood

    def setRewardDict(self, gameState, action):
        furthestFood = self.getFurthestFood(gameState)
        nextState = gameState.generateSuccessor(self.index, action)
        if nextState.getAgentPosition(self.index) == furthestFood:
            self.rewardDict[(gameState, action)] = 100

    def getReward(self, gameState, action):
        nextState = gameState.generateSuccessor(self.index, action)
        pos = nextState.getAgentPosition(self.index)
        foods = self.getFood(gameState)
        furtestFood = self.getFurthestFood(gameState)
        legalActions = gameState.getLegalActions(self.index)
        if action not in legalActions:
            return -10
        elif pos in foods:
            return 100
        else:
            return 0
            #return float(1) / self.getMazeDistance(pos, furtestFood)

    def epsilonAction(self, gameState):

        legalActions = gameState.getLegalActions(self.index)
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        else:
            return self.getBestAction(gameState)

    def chooseAction(self, gameState):
        #curState = gameState


        if util.flipCoin(self.epsilon):
            legalActions = gameState.getLegalActions(self.index)
            random.choice(legalActions)
        else:

            for i in range(50):
                #print i
                curState = gameState
                foods = self.getFood(curState).asList()
                while curState.getAgentPosition(self.index) not in foods:
                    curPos = curState.getAgentPosition(self.index)
                    legalActions = curState.getLegalActions(self.index)
                    legalActions.remove('Stop')
                    #action = random.choice(legalActions)
                    action = self.getBestAction(curState)
                    nextState = curState.generateSuccessor(self.index, action)
                    futureRewards = []
                    for nextAction in nextState.getLegalActions(self.index):
                        futureRewards.append(self.getQValue(nextState, nextAction))
                    QState = self.getReward(curState, action) + self.gamma * max(futureRewards)
                    self.brain[(curState, action)] = QState
                    curState = nextState
        return self.getBestAction(gameState)


































































