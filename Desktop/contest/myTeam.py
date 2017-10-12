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
from game import Directions, Actions
import game
from util import nearestPoint



#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='CasualTeam', second='DefensiveReflexAgent'):
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


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}


class QLearningAgent(ReflexCaptureAgent):

    def __init__(self, index, epsilon = 0.1, gamma = 0.9, alpha = 0.8):
        self.index = index
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.brain = util.Counter()
        self.observationHistory = []

    def enemyPosition(self, gameState):
        enemyPos = []
        for enemy in self.getOpponents(gameState):
            pos = gameState.getAgentPosition(enemy)
            if pos != None:
                enemyPos.append((enemy, pos))
        return enemyPos

    def enemyDistance(self, gameState):
        pos = self.enemyPosition(gameState)
        minDis = 9999
        if len(pos) > 0:
            myPos = gameState.getAgentPosition(self.index)
            for i, p in pos:
                dist = self.getMazeDistance(p, myPos)
                if dist < minDis:
                    minDis = dist
        if minDis == 9999:
            return -1
        return minDis





    def getFeatures(self, gameState, action):

        features = util.Counter()
        nextState = gameState.generateSuccessor(self.index, action)
        foods = self.getFood(gameState).asList()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [x for x in enemies if x.isPacman and x.getPosition() != None]
        pos = gameState.getAgentPosition(self.index)


        features['successorScore'] = -len(foodList)


        features['numInvaders'] = len(invaders)

        if len(invaders) > 0:
            dists = [self.getMazeDistance(pos, x.getPosition()) for x in invaders]
            features['invaderDis'] = min(dists)


        enemyDis = self.enemyDistance(gameState)
        if enemyDis < 6 and not enemyDis is None:
            features['danger'] = 1
        else:
            features['danger'] = 0

        if action == 'Stop':
            features['stop'] = 1


        capsules = self.getCapsules(gameState)
        if len(capsules) > 0:
            minCapsuleDis = min([self.getMazeDistance(pos, capsule) for capsule in capsules])
        else:
            minCapsuleDis = 0.1
        features['capsuleDis'] = 1 / minCapsuleDis




        # Compute distance to the nearest food

        if len(foodList) > 0: # This should always be True,  but better safe than sorry
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            nextState = gameState.generateSuccessor(self.index, action)
            if nextState.getAgentPosition(self.index) in foods:
                features['distanceToFood'] = 1

            else:
                features['distanceToFood'] = minDistance

        return features

    

    def getWeights(self, gameState, action):
        if self.enemyDistance(gameState) < 6 and not self.enemyDistance(gameState) is None:
            return {'successorScore': 2, 'distanceToFood': -1000, 'danger': -400, 'capsuleDis': 100, 'stop': -2000, 'numInvaders': -20}

        return {'successorScore': 100, 'distanceToFood': -1, 'danger': -1000, 'capsuleDis': 100, 'stop': -2000}



    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)




    def getQValue(self, gameState, action):

        #features = self.getFeatures(gameState,action)

        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        res = features * weights

        return res

        #qVal = features * weights
        #return features * weights



    def getNormalValue(self, gameState, action):

        features = self.getFeatures(gameState,action)
        weights = self.getWeights(gameState, action)
        return features * weights


    def computeValueFromQValues(self, gameState):
        maxQvalue = float('-inf')
        for action in gameState.getLegalActions(self.index):
            maxQvalue = max(maxQvalue, self.getQValue(gameState, action))
        return maxQvalue if maxQvalue != float('-inf') else 0.0

    def computeValueFromNormalValues(self, gameState):
        maxNormalvalue = float('-inf')
        for action in gameState.getLegalActions(self.index):
            maxNormalvalue = max(maxNormalvalue, self.getNormalValue(gameState, action))
        return maxNormalvalue if maxNormalvalue != float('-inf') else 0.0

    def getNormalAction(self, gameState):
        if len(gameState.getLegalActions(self.index)) == 0:
            return None

        bestQValue = self.computeValueFromNormalValues(gameState)
        bestActions = []
        for action in gameState.getLegalActions(self.index):
            if bestQValue == self.getNormalValue(gameState, action):
                bestActions.append(action)
        return random.choice(bestActions)

    def getBestAction(self, gameState):
        if len(gameState.getLegalActions(self.index)) == 0:
            return None





        for action in gameState.getLegalActions(self.index):
            feature = self.getFeatures(gameState, action)
            weight = self.getWeights(gameState, action)
            val = self.getQValue(gameState, action)
            enemPos = self.enemyPosition(gameState)
            enemyDis = self.enemyDistance(gameState)
            foods = self.getFood(gameState).asList()
            foodslen = len(foods)
            curPos = gameState.getAgentPosition(self.index)
            isFood = gameState.getAgentPosition(self.index) in foods

            x = 1





        bestQValue = self.computeValueFromQValues(gameState)
        bestActions = []
        for action in gameState.getLegalActions(self.index):
            valTmp = self.getQValue(gameState, action)
            nextState = gameState.generateSuccessor(self.index, action)
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
            #print feature, value
            self.brain[feature] += self.alpha * difference * features[feature]
        newBrain = self.brain




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
            return 10
        else:
            return -1
            #return float(1) / self.getMazeDistance(pos, nearestFood)

    def containFoodIn3(self, gameState):
        foods = self.getFood(gameState).asList()
        curState = gameState
        curPos = curState.getAgentPosition(self.index)
        for food in foods:
            if self.getMazeDistance(food, curPos) < 3:
                return True
        return False

    def nearestFoodIn3(self, gameState):
        foods = self.getFood(gameState).asList()
        for food in foods:
            if self.getMazeDistance(food, gameState.getAgentPosition(self.index)) < 3:
                return food

    def inRange3(self, gameState):
        pos_x, pos_y = gameState.getAgentPosition(self.index)
        res = []
        for x in [-3, -2, -1, 0, 1, 2, 3]:
            for y in [-3, -2, -1, 0, 1, 2, 3]:
                res.append((pos_x + x, pos_y + y))
        return res



    def chooseAction(self, gameState):
        legalActions = gameState.getLegalActions(self.index)
        #legalActions.remove('Stop')
        enemies = [gameState.getAgentState(i).getPosition() for i in self.getOpponents(gameState)]
        if gameState.getAgentState(self.index).numCarrying > 5:
            bestDist = 9999
            actions = gameState.getLegalActions(self.index)
            for action in actions:

                successor = self.getSuccessor(gameState, action)
                if successor.getAgentPosition(self.index) in enemies:
                    continue

                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction
        elif util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        elif not self.containFoodIn3(gameState):
           # print self.enemyDistance(gameState)
            #print self.containFoodIn5(gameState)
            return self.getNormalAction(gameState)
        else:
            for i in range(1):
                curState = gameState
                curPos = curState.getAgentPosition(self.index)
                nearestFood = self.nearestFoodIn3(gameState)
                rangeIn3 = self.inRange3(curState)
                count = 0

                while curState.getAgentPosition(self.index) != nearestFood:
                    #curPos = curState.getAgentPosition(self.index)
                    legalActions = curState.getLegalActions(self.index)
                    #legalActions.remove('Stop')
                    action = random.choice(legalActions)
                    while curState.generateSuccessor(self.index, action).getAgentPosition(self.index) not in rangeIn3:
                        action = random.choice(legalActions)

                    nextState = curState.generateSuccessor(self.index, action)

                    reward = self.getReward(curState, action)
                    self.update(curState, action, nextState, reward)

                    count += 1
                    curState = nextState
                newBrain = self.brain

            return self.getBestAction(gameState)


class MonteCarloAgent(ReflexCaptureAgent):

    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.inactiveTime = 0
        self.enemyFoodLeft = '+inf'

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)


    def enemyPosition(self, gameState):
        enemyPos = []
        for enemy in self.getOpponents(gameState):
            pos = gameState.getAgentPosition(enemy)
            if pos != None:
                enemyPos.append((enemy, pos))
        return enemyPos

    def enemyDistance(self, gameState):
        pos = self.enemyPosition(gameState)
        minDis = 9999
        if len(pos) > 0:
            myPos = gameState.getAgentPosition(self.index)
            for i, p in pos:
                dist = self.getMazeDistance(p, myPos)
                if dist < minDis:
                    minDis = dist
        if minDis == 9999:
            return -1
        return minDis



    def getFeatures(self, gameState, action):
        features = util.Counter()
        nextState = gameState.generateSuccessor(self.index, action)
        foods = self.getFood(gameState).asList()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        enemyDis = self.enemyDistance(nextState)
        invaders = [x for x in enemies if x.isPacman and x.getPosition() != None]
        pos = gameState.getAgentPosition(self.index)

        features['successorScore'] = self.getScore(nextState)

        if enemyDis < 6 and enemyDis != -1:
            features['danger'] = 1

        else:
            features['danger'] = 0
        if enemyDis < 4 and enemyDis != -1 and not gameState.getAgentState(self.index).isPacman:
            features['danger'] = -1



        if len(foodList) > 0:
            myPos = successor.getAgentState(self.index).getPosition()
            #minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            nextState = gameState.generateSuccessor(self.index, action)
            features['distanceToFood'] = minDistance


        if gameState.getAgentState(self.index).numCarrying > 3:
            features['disToHome'] = self.getMazeDistance(pos, gameState.getInitialAgentPosition(self.index))
        else:
            features['disToHome'] = 0

        inRange = filter(lambda x: not x.isPacman and x.getPosition() != None, enemies)
        if len(inRange) > 0:
            positions = [agent.getPosition() for agent in inRange]
            closest = min(positions, key=lambda x: self.getMazeDistance(myPos, x))
            closestDist = self.getMazeDistance(myPos, closest)
            if closestDist <= 5:
                features['distanceToGhost'] = closestDist

        features['isPacman'] = 1 if successor.getAgentState(self.index).isPacman else 0

        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1


        return features

    def getWeights(self, gameState, action):
        #return {'successorScore': 100, 'distanceToFood': -1, 'danger': -1000, 'capsuleDis': 1, 'disToHome': -1}
        if self.inactiveTime > 50:
            return {'successorScore': 200, 'distanceToFood': -20, 'distanceToGhost': 2, 'isPacman': 1000}
        return {'successorScore': 200, 'distanceToFood': -5, 'distanceToGhost': 2, 'isPacman': 0}

    def calculateValue(self, gameState, action):
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getNothing(self, gameState, action, depth):
        if depth == 0:
            return False
        curScore = self.getScore(gameState)
        nextState = gameState.generateSuccessor(self.index, action)
        nextScore = self.getScore(nextState)
        if nextScore > curScore:
            return False
        actions = nextState.getLegalActions(self.index)
        actions.remove('Stop')
        revDirection = Directions.REVERSE[nextState.getAgentState(self.index).configuration.direction]
        if revDirection in actions:
            actions.remove(revDirection)
        if len(action) == 0:
            return True
        for action in actions:
            if not self.getNothing(nextState, action, depth - 1):
                return False
        return True


    def monteCarloWalk(self, depth, gameState):
        #res = 0
        curState = gameState.deepCopy()
        while depth > 0:
            actions = curState.getLegalActions(self.index)
            actions.remove('Stop')
            curDirection = curState.getAgentState(self.index).configuration.direction
            revDirection = Directions.REVERSE[curState.getAgentState(self.index).configuration.direction]
            if revDirection in actions and len(actions) > 1:
                actions.remove(revDirection)
            action = random.choice(actions)
            #res += self.calculateValue(curState, action)
            curState = curState.generateSuccessor(self.index, action)
            depth -= 1
        return self.calculateValue(curState, Directions.STOP)
        #return res

    def chooseAction(self, gameState):


        curFoodLeft = len(self.getFood(gameState).asList())
        if self.enemyFoodLeft == curFoodLeft:
            self.inactiveTime += 1
        else:
            self.enemyFoodLeft = curFoodLeft
            self.inactiveTime = 0
        if gameState.getInitialAgentPosition(self.index) == gameState.getAgentPosition(self.index):
            self.inactiveTime = 0

        '''
        actionsTmp = gameState.getLegalActions(self.index)
        actionsTmp.remove('Stop')
        actions = []
        for a in actionsTmp:
            if not self.getNothing(gameState, a, 5):
                actions.append(a)
        if len(actions) == 0:
            actions = actionsTmp
        '''
        actions = gameState.getLegalActions(self.index)
        actions.remove('Stop')
        values = []
        for action in actions:
            nextState = gameState.generateSuccessor(self.index, action)
            value = 0
            for i in range(10):
                #depth = random.randint(3, 10)
                value += self.monteCarloWalk(2, nextState)


            features = self.getFeatures(gameState, action)
            weights = self.getWeights(gameState, action)



            values.append((value, action))

        maxVal = max(values)
        res = []
        for value in values:
            if value == maxVal:
                res.append(value[1])
        actTime = self.inactiveTime

        return random.choice(res)


class Caesar(ReflexCaptureAgent):
    def getFeatures(self, state, action):
        food = self.getFood(state)
        foodList = food.asList()
        walls = state.getWalls()
        isPacman = self.getSuccessor(state, action).getAgentState(self.index).isPacman

        # Zone of the board agent is primarily responsible for
        zone = (self.index - self.index % 2) / 2

        teammates = [state.getAgentState(i).getPosition() for i in self.getTeam(state)]
        opponents = [state.getAgentState(i) for i in self.getOpponents(state)]
        chasers = [a for a in opponents if not (a.isPacman) and a.getPosition() != None]
        prey = [a for a in opponents if a.isPacman and a.getPosition() != None]

        features = util.Counter()
        if action == Directions.STOP:
            features["stopped"] = 1.0
        # compute the location of pacman after he takes the action
        x, y = state.getAgentState(self.index).getPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        for g in chasers:
            if (next_x, next_y) == g.getPosition():
                if g.scaredTimer > 0:
                    features["eats-ghost"] += 1
                    features["eats-food"] += 2
                else:
                    features["#-of-dangerous-ghosts-1-step-away"] = 1
                    features["#-of-harmless-ghosts-1-step-away"] = 0
            elif (next_x, next_y) in Actions.getLegalNeighbors(g.getPosition(), walls):
                if g.scaredTimer > 0:
                    features["#-of-harmless-ghosts-1-step-away"] += 1
                elif isPacman:
                    features["#-of-dangerous-ghosts-1-step-away"] += 1
                    features["#-of-harmless-ghosts-1-step-away"] = 0
        if state.getAgentState(self.index).scaredTimer == 0:
            for g in prey:
                if (next_x, next_y) == g.getPosition:
                    features["eats-invader"] = 1
                elif (next_x, next_y) in Actions.getLegalNeighbors(g.getPosition(), walls):
                    features["invaders-1-step-away"] += 1
        else:
            for g in opponents:
                if g.getPosition() != None:
                    if (next_x, next_y) == g.getPosition:
                        features["eats-invader"] = -10
                    elif (next_x, next_y) in Actions.getLegalNeighbors(g.getPosition(), walls):
                        features["invaders-1-step-away"] += -10

        for capsule_x, capsule_y in state.getCapsules():
            if next_x == capsule_x and next_y == capsule_y and isPacman:
                features["eats-capsules"] = 1.0
        if not features["#-of-dangerous-ghosts-1-step-away"]:
            if food[next_x][next_y]:
                features["eats-food"] = 1.0
            if len(foodList) > 0:  # This should always be True,  but better safe than sorry
                myFood = []
                for food in foodList:
                    food_x, food_y = food
                    if (food_y > zone * walls.height / 3 and food_y < (zone + 1) * walls.height / 3):
                        myFood.append(food)
                if len(myFood) == 0:
                    myFood = foodList
                myMinDist = min([self.getMazeDistance((next_x, next_y), food) for food in myFood])
                if myMinDist is not None:
                    features["closest-food"] = float(myMinDist) / (walls.width * walls.height)

        features.divideAll(10.0)

        return features

    def getWeights(self, gameState, action):
        return {'eats-invader': 5, 'invaders-1-step-away': 0, 'teammateDist': 1.5, 'closest-food': -1,
                'eats-capsules': 10.0, '#-of-dangerous-ghosts-1-step-away': -20, 'eats-ghost': 1.0,
                '#-of-harmless-ghosts-1-step-away': 0.1, 'stopped': -5, 'eats-food': 1}


class CasualTeam(ReflexCaptureAgent):


    def getFeatures(self, gameState, action):
        foods = self.getFood(gameState)
        foodList = foods.asList()
        walls = gameState.getWalls()
        isPacman = self.getSuccessor(gameState, action).getAgentState(self.index).isPacman
        zone = (self.index - self.index % 2) / 2

        opponents = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        chasers = [a for a in opponents if not (a.isPacman) and a.getPosition() != None]
        invaders = [i for i in opponents if i.isPacman and i.getPosition() != None]

        features = util.Counter()

        if action == Directions.STOP:
            features['Stop'] = 1
        nextState = gameState.generateSuccessor(self.index, action)
        nextPos = nextState.getAgentPosition(self.index)
        nextLegalActions = nextState.getLegalActions(self.index)
        nextLegalActions.remove('Stop')
        for chaser in chasers:
            rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
            backPos = gameState.generateSuccessor(self.index, rev).getAgentPosition(self.index)
            actionsTmp = nextState.getLegalActions(self.index)
            actionsTmp.remove('Stop')

            #if backPos == chaser.getPosition and len(actionsTmp) == 1:
              #  features['DeadEnd'] = 1

            if nextPos == chaser.getPosition():
                if chaser.scaredTimer > 0:
                    features['EatGhost'] += 1
                    features['EatFood'] += 2
                else:
                    features['DangerousGhost'] = 1
                    features['SafeGhost'] = 0
            elif nextPos in Actions.getLegalNeighbors(chaser.getPosition(), walls):

                if chaser.scaredTimer > 0:
                    features['SafeGhost'] += 1
                elif isPacman:
                    features['SafeGhost'] = 0
                    features['DangerousGhost'] += 2

        if gameState.getAgentState(self.index).scaredTimer == 0:
            for invader in invaders:
                if nextPos == invader.getPosition:
                    features['EatInvader'] = 1
                elif nextPos in Actions.getLegalNeighbors(invader.getPosition(), walls):
                    features['OneStepInvader'] += 1
        else:
            for invader in invaders:
                if nextPos == invader.getPosition:
                    features['EatInvader'] = -10
                elif nextPos in Actions.getLegalNeighbors(invader.getPosition(), walls):
                    features['OneStepInvader'] += -10

        for capsule in self.getCapsules(gameState):
            if nextPos == capsule:
                features['EatCapsule'] = 1

        if not features['DangerousGhost']:
            if nextPos in foods:
                features['EatFood'] = 1.0
            if len(foodList) > 0:  # This should always be True,  but better safe than sorry
                myFood = []
                for food in foodList:
                    food_x, food_y = food
                    if (food_y > zone * walls.height / 3 and food_y < (zone + 1) * walls.height / 3):
                        myFood.append(food)
                if len(myFood) == 0:
                    myFood = foodList
                myMinDist = min([self.getMazeDistance(nextPos, food) for food in myFood])
                if myMinDist is not None:
                    features["ClosestFood"] = float(myMinDist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features

    def getWeights(self, gameState, action):
        return {'Stop': -5, 'EatGhost': 1.0, 'DeadEnd': -2, 'EatFood': 1, 'DangerousGhost': -20, 'SafeGhost': 0.1,
                'EatInvader': 5, 'OneStepInvader': 0, 'EatCapsule': 10, 'ClosestFood': -1}
    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]

        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        foodLeft = len(self.getFood(gameState).asList())

        #print gameState.getAgentState(self.index).numCarrying
        #carry = random.randint(1, 5)

        if gameState.getAgentState(self.index).numCarrying > 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        return random.choice(bestActions)























































































