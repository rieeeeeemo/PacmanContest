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
from baselineTeam import ReflexCaptureAgent
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

class QLearningAgent(ReflexCaptureAgent):

    def __init__(self, index, epsilon = 0.1, gamma = 0.9, alpha = 0.8):
        self.index = index
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.brain = util.Counter()
        self.observationHistory = []



    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()    
        features['successorScore'] = -len(foodList)#self.getScore(successor)

        # Compute distance to the nearest food

        if len(foodList) > 0: # This should always be True,  but better safe than sorry
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance
        return features

    def getWeights(self, gameState, action):
        return {'successorScore': 50, 'distanceToFood': -10}


    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)


    def getQValue(self, gameState, action):
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def computeValueFromQValues(self, gameState):
        maxQvalue = float('-inf')
        for action in gameState.getLegalActions(self.index):
            maxQvalue = max(maxQvalue, self.getQValue(gameState, action))
        return maxQvalue if maxQvalue != float('-inf') else 0.0



    def getBestAction(self, gameState):
        if len(gameState.getLegalActions(self.index)) == 0:
            return None

        bestQValue = self.computeValueFromQValues(gameState)
        bestActions = []
        for action in gameState.getLegalActions(self.index):
            if bestQValue == self.getQValue(gameState, action):
                bestActions.append(action)
        return random.choice(bestActions)

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
        difference = (reward + self.gamma * self.computeValueFromQValues(nextState)) - self.getQValue(gameState, action)
        features = self.getFeatures(gameState, action)
        for feature, value in features.items():
            self.brain[features] += self.alpha * difference * features[feature]

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
        
    def getNearestFood(self, gameState):
        nearestFood = None
        foods = self.getFood(gameState).asList()
        curPos = gameState.getAgentPosition(self.index)
        if len(foods) == 0:
            return (0, 0)
        minDis = 9999
        for food in foods:
            if self.getMazeDistance(curPos, food) <= minDis:
                minDis = self.getMazeDistance(curPos, food)
                nearestFood = food
        return nearestFood


    def getReward(self, gameState, action):
        nextState = gameState.generateSuccessor(self.index, action)
        pos = nextState.getAgentPosition(self.index)
        foods = self.getFood(gameState)
        furtestFood = self.getFurthestFood(gameState)
        nearestFood = self.getNearestFood(gameState)
        legalActions = gameState.getLegalActions(self.index)
        enemiesPos = [nextState.getAgentState(i).getPosition() for i in self.getOpponents(nextState)]

        if pos in enemiesPos:
            return -100
        if action not in legalActions:
            return -10
        elif pos in foods:
            return 100
        else:
            return 0
            #return float(1) / self.getMazeDistance(pos, nearestFood)

    '''

    def chooseAction(self, gameState):
        curPos = gameState.getAgentPosition(self.index)
        nearestfood=self.getNearestFood(gameState)
        self.disTmp = self.getMazeDistance(curPos,nearestfood)
        print self.getMazeDistance(curPos,nearestfood)
        if self.disTmp > 5:
        
            actions = gameState.getLegalActions(self.index)

            # You can profile your evaluation time by uncommenting these lines
            # start = time.time()
            values = [self.evaluate(gameState, a) for a in actions]
            # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

            maxValue = max(values)
        
            bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        

            foodLeft = len(self.getFood(gameState).asList())

            if foodLeft <= 2:
                bestDist = 9999
                for action in actions:
                    successor = self.getSuccessor(gameState, action)
                    pos2 = successor.getAgentPosition(self.index)
                    dist = self.getMazeDistance(self.start,pos2)
                    if dist < bestDist:
                        bestAction = action
                        bestDist = dist
                return bestAction

            return random.choice(bestActions)


        else:
            if util.flipCoin(self.epsilon):
                legalActions = gameState.getLegalActions(self.index)
                return random.choice(legalActions)
            else:

                for i in range(50):
                    #print i
                    curState = gameState
                    foods = self.getFood(curState).asList()
                    while curState.getAgentPosition(self.index) not in foods:
                        curPos = curState.getAgentPosition(self.index)
                        legalActions = curState.getLegalActions(self.index)
                        legalActions.remove('Stop')

                        #action = self.getBestAction(curState)
                        action = random.choice(legalActions)
                        nextState = curState.generateSuccessor(self.index, action)
                        futureRewards = []
                        for nextAction in nextState.getLegalActions(self.index):
                            futureRewards.append(self.getQValue(nextState, nextAction))
                        QState = self.getQValue(gameState, action) + self.alpha * (self.getReward(curState, action) +
                                                                                   self.gamma * max(futureRewards) -
                                                                                   self.getQValue(gameState, action))
                        self.brain[(curState, action)] = QState
                        curState = nextState
                        #print curState
            return self.getBestAction(gameState)
    '''

    def chooseAction(self, gameState):
        legalActions = gameState.getLegalActions(self.index)
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        else:
            return self.getBestAction(gameState)


































































