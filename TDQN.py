import numpy as np
import math
import random
import copy
import datetime


# we'll be using a deque to store our observation space
from collections import deque
from tqdm import tqdm
from matplotlib import pyplot as plt

#we need pytorch to design our model architecture
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from tradingPerformance import PerformanceEstimator
from dataAugmentation import DataAugmentation
from tradingEnv import TradingEnv

#we will now declare some of the global variables related to our model and other fucntions that we'll be using through this program

#parameters for our model
gamma = 0.4
learningRate = 0.0001
targetNetworkUpdate = 1000
learningUpdatePeriod = 1

numberOfNeurons = 512
dropout = 0.2
alpha = 0.1

#for our experience replay
capacity = 100000
batchSize = 32
experiencesRequired = 1000


# we'll be using an Epsilon-Greedy exploration technique
epsilonStart = 1.0
epsilonEnd = 0.01
epsilonDecay = 10000


filterOrder = 5

#we also plan to clip the gradient and rewards for better stabalization
#If rewards vary too much, the agent over-focuses on rare large rewards instead of learning a stable policy.
#in deep NN's gradient can grow exponentially. If gradients become too large, weights update too aggressively, destabilizing training.
gradientClipping = 1
rewardClipping = 1

#L2 Regularization 
L2Factor = 0.000001

#we won't be using hardware acceleration (CUDA) in our case (this'll can be changed in the TDQN class is torch.device detects a GPU)
GPUNumber = 0



#thsi class will help us to handle(store, retrive and reset) the experiene replay for our the DQN RL algorithm
class ReplayMemory:

    def __init__(self, capacity=capacity):
        self.memory = deque(maxlen=capacity)
    

    def push(self, state, action, reward, nextState, done):
        self.memory.append((state, action, reward, nextState, done))


    def sample(self, batchSize):
        state, action, reward, nextState, done = zip(*random.sample(self.memory, batchSize))
        return state, action, reward, nextState, done


    def __len__(self):
        return len(self.memory)


    def reset(self):
        self.memory = deque(maxlen=capacity)


#we will now implement our DQN Architecture using pytorch
# we'll be using fully connected, dropout(to prevent overfitting) and batch normalized(normalizes the activations of neurons within each mini-batch) layers in teh below format
'''
I'll now give a basic structure of the model that we'll try to implement belwo:
Input → FC1 → BN1 → LeakyReLU → Dropout1
       → FC2 → BN2 → LeakyReLU → Dropout2
       → FC3 → BN3 → LeakyReLU → Dropout3
       → FC4 → BN4 → LeakyReLU → Dropout4
       → FC5 → Output
'''

class DQN(nn.Module):

    def __init__(self, numberOfInputs, numberOfOutputs, numberOfNeurons=numberOfNeurons, dropout=dropout):
        # Calling the constructor of the parent class(Pytorch nn.Module)
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(numberOfInputs, numberOfNeurons)
        self.fc2 = nn.Linear(numberOfNeurons, numberOfNeurons)
        self.fc3 = nn.Linear(numberOfNeurons, numberOfNeurons)
        self.fc4 = nn.Linear(numberOfNeurons, numberOfNeurons)
        self.fc5 = nn.Linear(numberOfNeurons, numberOfOutputs)

        self.bn1 = nn.BatchNorm1d(numberOfNeurons)
        self.bn2 = nn.BatchNorm1d(numberOfNeurons)
        self.bn3 = nn.BatchNorm1d(numberOfNeurons)
        self.bn4 = nn.BatchNorm1d(numberOfNeurons)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        # Instead of random initialization, we'll use Xavier initialization for the entire neural network
        # this helps helps neural networks train faster and avoid vanishing/exploding gradients.
        # here weight are basically coming from a normnal dist. N(0, 2/(n_in + n_out)) where n_in is the number of input neurons and n_out is the number of output neurons
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.xavier_uniform_(self.fc4.weight)
        torch.nn.init.xavier_uniform_(self.fc5.weight)

    
    #note: we're using the leaky relu activation function instead of the relu activation function
    def forward(self, input):
        x = self.dropout1(F.leaky_relu(self.bn1(self.fc1(input))))
        x = self.dropout2(F.leaky_relu(self.bn2(self.fc2(x))))
        x = self.dropout3(F.leaky_relu(self.bn3(self.fc3(x))))
        x = self.dropout4(F.leaky_relu(self.bn4(self.fc4(x))))
        output = self.fc5(x)
        return output


#we'll now implement our trading agendt
class TDQN:
    def __init__(self, observationSpace, actionSpace, numberOfNeurons=numberOfNeurons, dropout=dropout, 
                 gamma=gamma, learningRate=learningRate, targetNetworkUpdate=targetNetworkUpdate,
                 epsilonStart=epsilonStart, epsilonEnd=epsilonEnd, epsilonDecay=epsilonDecay,
                 capacity=capacity, batchSize=batchSize):
        random.seed(0)

        self.device = torch.device('cuda:'+str(GPUNumber) if torch.cuda.is_available() else 'cpu')

        self.gamma = gamma
        self.learningRate = learningRate
        self.targetNetworkUpdate = targetNetworkUpdate
        self.observationSpace = observationSpace
        self.actionSpace = actionSpace

        # for experience reply
        self.capacity = capacity
        self.batchSize = batchSize
        self.replayMemory = ReplayMemory(capacity)

        # We'll be using two NN's -> policy and target
        # Policy Network (Active Network) – Learns Q-values and is used for action selection.
        # Target Network (Fixed Network) – Provides stable Q-value targets to prevent training instability.
        self.policyNetwork = DQN(observationSpace, actionSpace, numberOfNeurons, dropout).to(self.device)
        self.targetNetwork = DQN(observationSpace, actionSpace, numberOfNeurons, dropout).to(self.device)
        self.targetNetwork.load_state_dict(self.policyNetwork.state_dict())
        self.policyNetwork.eval()
        self.targetNetwork.eval()

        self.optimizer = optim.Adam(self.policyNetwork.parameters(), lr=learningRate, weight_decay=L2Factor)

        self.epsilonValue = lambda iteration: epsilonEnd + (epsilonStart - epsilonEnd) * math.exp(-1 * iteration / epsilonDecay)
        
        self.iterations = 0

        # Initialization of the tensorboard writer
        self.writer = SummaryWriter('runs/' + datetime.datetime.now().strftime("%d/%m/%Y-%H:%M:%S"))


    #we now need normalized coeeficients for the input data -> this'll make our operations much for efficient and solve the problem of different size of valyes of each feature
    def getNormalizationCoefficients(self, tradingEnv):
        tradingData = tradingEnv.data
        closePrices = tradingData['Close'].tolist()
        lowPrices = tradingData['Low'].tolist()
        highPrices = tradingData['High'].tolist()
        volumes = tradingData['Volume'].tolist()

        coefficients = []
        margin = 1

        returns = [abs((closePrices[i]-closePrices[i-1])/closePrices[i-1]) for i in range(1, len(closePrices))]
        coeffs = (0, np.max(returns)*margin)
        coefficients.append(coeffs)

        deltaPrice = [abs(highPrices[i]-lowPrices[i]) for i in range(len(lowPrices))]
        coeffs = (0, np.max(deltaPrice)*margin)
        coefficients.append(coeffs)

        coeffs = (0, 1)
        coefficients.append(coeffs)

        coeffs = (np.min(volumes)/margin, np.max(volumes)*margin)
        coefficients.append(coeffs)
        
        return coefficients


#we will now approproiately format and nornmalize the inout data
    def processState(self, state, coefficients):
        closePrices = [state[0][i] for i in range(len(state[0]))]
        lowPrices = [state[1][i] for i in range(len(state[1]))]
        highPrices = [state[2][i] for i in range(len(state[2]))]
        volumes = [state[3][i] for i in range(len(state[3]))]

        returns = [(closePrices[i]-closePrices[i-1])/closePrices[i-1] for i in range(1, len(closePrices))]
        if coefficients[0][0] != coefficients[0][1]:
            state[0] = [((x - coefficients[0][0])/(coefficients[0][1] - coefficients[0][0])) for x in returns]
        else:
            state[0] = [0 for x in returns]

        deltaPrice = [abs(highPrices[i]-lowPrices[i]) for i in range(1, len(lowPrices))]
        if coefficients[1][0] != coefficients[1][1]:
            state[1] = [((x - coefficients[1][0])/(coefficients[1][1] - coefficients[1][0])) for x in deltaPrice]
        else:
            state[1] = [0 for x in deltaPrice]

        closePricePosition = []
        for i in range(1, len(closePrices)):
            deltaPrice = abs(highPrices[i]-lowPrices[i])
            if deltaPrice != 0:
                item = abs(closePrices[i]-lowPrices[i])/deltaPrice
            else:
                item = 0.5
            closePricePosition.append(item)
        if coefficients[2][0] != coefficients[2][1]:
            state[2] = [((x - coefficients[2][0])/(coefficients[2][1] - coefficients[2][0])) for x in closePricePosition]
        else:
            state[2] = [0.5 for x in closePricePosition]

        volumes = [volumes[i] for i in range(1, len(volumes))]
        if coefficients[3][0] != coefficients[3][1]:
            state[3] = [((x - coefficients[3][0])/(coefficients[3][1] - coefficients[3][0])) for x in volumes]
        else:
            state[3] = [0 for x in volumes]
        
        state = [item for sublist in state for item in sublist]

        return state

    
    def processReward(self, reward):
        return np.clip(reward, -rewardClipping, rewardClipping)
 

#we'll now setup the updating frew for our Target network
    def updateTargetNetwork(self):
        if(self.iterations % targetNetworkUpdate == 0):
            self.targetNetwork.load_state_dict(self.policyNetwork.state_dict())


    def chooseAction(self, state):
        with torch.no_grad():
            tensorState = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
            QValues = self.policyNetwork(tensorState).squeeze(0)
            Q, action = QValues.max(0)
            action = action.item()
            Q = Q.item()
            QValues = QValues.cpu().numpy()
            return action, Q, QValues

    
    def chooseActionEpsilonGreedy(self, state, previousAction):
        # Our exploitation strategy is to use the RL policy
        if(random.random() > self.epsilonValue(self.iterations)):
            # We'll use the Sticky action which is an RL generalization mechanism
            if(random.random() > alpha):
                action, Q, QValues = self.chooseAction(state)
            else:
                action = previousAction
                Q = 0
                QValues = [0, 0]

        # We'll randomly choose our actions for our exploration strategy
        else:
            action = random.randrange(self.actionSpace)
            Q = 0
            QValues = [0, 0]
        
        self.iterations += 1

        return action, Q, QValues
    
#we'll now bascially sample a batch of our past memory and learn from it by updating our RL policy
#unsqueeze(i) function adds a new dimension to a PyTorch tensor at the i'th index. Neural networks in PyTorch expect batched inputs
    def learning(self, batchSize=batchSize):
        if (len(self.replayMemory) >= batchSize):

            self.policyNetwork.train()
            state, action, reward, nextState, done = self.replayMemory.sample(batchSize)

            # Initialization of Pytorch tensors for the RL experience elements (state, action, reward, next state, done)
            state = torch.tensor(state, dtype=torch.float, device=self.device)
            action = torch.tensor(action, dtype=torch.long, device=self.device)
            reward = torch.tensor(reward, dtype=torch.float, device=self.device)
            nextState = torch.tensor(nextState, dtype=torch.float, device=self.device)
            done = torch.tensor(done, dtype=torch.float, device=self.device)


            currentQValues = self.policyNetwork(state).gather(1, action.unsqueeze(1)).squeeze(1)

            # Compute the next Q values returned by the target network
            with torch.no_grad():
                nextActions = torch.max(self.policyNetwork(nextState), 1)[1]
                nextQValues = self.targetNetwork(nextState).gather(1, nextActions.unsqueeze(1)).squeeze(1)
                expectedQValues = reward + gamma * nextQValues * (1 - done)

            # We're using Huber loss in our implementation -> It is a combination of Mean Squared Error (MSE) for small differences and Mean Absolute Error (MAE) for large differences.
            loss = F.smooth_l1_loss(currentQValues, expectedQValues)

            self.optimizer.zero_grad() # Clears previous gradients.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policyNetwork.parameters(), gradientClipping)

            # Perform the Deep Neural Network optimization
            self.optimizer.step() #Updates network weights.
            self.updateTargetNetwork()
            self.policyNetwork.eval()
            #in eval mode -> disables the dropout and batch normalization updates. Ensures the networks behave deterministically


    def training(self, trainingEnv, trainingParameters=[],
                 verbose=False, rendering=False, plotTraining=False, showPerformance=False):
        
        #We need to generate multiple variations of the training environment to improve generalization.
        dataAugmentation = DataAugmentation()
        trainingEnvList = dataAugmentation.generate(trainingEnv)

        if plotTraining:
            # Training performance
            performanceTrain = []
            score = np.zeros((len(trainingEnvList), trainingParameters[0]))

            # Testing performance
            marketSymbol = trainingEnv.marketSymbol
            startingDate = trainingEnv.endingDate
            endingDate = '2025-1-1'
            money = trainingEnv.data['Money'][0]
            stateLength = trainingEnv.stateLength
            transactionCosts = trainingEnv.transactionCosts
            testingEnv = TradingEnv(marketSymbol, startingDate, endingDate, money, stateLength, transactionCosts)
            performanceTest = []

        try:
            if verbose:
                print("Training progression (hardware selected => " + str(self.device) + "):")

            # for each training episode
            for episode in tqdm(range(trainingParameters[0]), disable=not(verbose)):
                # each epoch -> training on the entire set of training environments
                for i in range(len(trainingEnvList)): #This'll reinitialize the states for each augmented environment. Env
                    coefficients = self.getNormalizationCoefficients(trainingEnvList[i])
                    trainingEnvList[i].reset()
                    startingPoint = random.randrange(len(trainingEnvList[i].data.index))
                    trainingEnvList[i].setStartingPoint(startingPoint)
                    state = self.processState(trainingEnvList[i].state, coefficients)
                    previousAction = 0
                    done = 0
                    stepsCounter = 0

                    if plotTraining:
                        totalReward = 0

                    while done == 0:
                        action, _, _ = self.chooseActionEpsilonGreedy(state, previousAction)
                        nextState, reward, done, info = trainingEnvList[i].step(action)
                        reward = self.processReward(reward)
                        nextState = self.processState(nextState, coefficients)
                        self.replayMemory.push(state, action, reward, nextState, done)

                        # We store an alternative action to improve exploration.
                        otherAction = int(not bool(action))
                        otherReward = self.processReward(info['Reward'])
                        otherNextState = self.processState(info['State'], coefficients)
                        otherDone = info['Done']
                        self.replayMemory.push(state, otherAction, otherReward, otherNextState, otherDone)

                        stepsCounter += 1
                        if stepsCounter == learningUpdatePeriod:
                            self.learning()
                            stepsCounter = 0

                        state = nextState
                        previousAction = action

                        if plotTraining:
                            totalReward += reward
                    
                    if plotTraining:
                        score[i][episode] = totalReward
                
                if plotTraining:
                    # Training set performance
                    trainingEnv = self.testing(trainingEnv, trainingEnv)
                    analyser = PerformanceEstimator(trainingEnv.data)
                    performance = analyser.computeSharpeRatio()
                    performanceTrain.append(performance)
                    self.writer.add_scalar('Training performance (Sharpe Ratio)', performance, episode)
                    trainingEnv.reset()
                    # Testing set performance
                    testingEnv = self.testing(trainingEnv, testingEnv)
                    analyser = PerformanceEstimator(testingEnv.data)
                    performance = analyser.computeSharpeRatio()
                    performanceTest.append(performance)
                    self.writer.add_scalar('Testing performance (Sharpe Ratio)', performance, episode)
                    testingEnv.reset()
        
        except KeyboardInterrupt:
            print()
            print("WARNING: Training prematurely interrupted...")
            print()
            self.policyNetwork.eval()

        trainingEnv = self.testing(trainingEnv, trainingEnv)

        if rendering:
            trainingEnv.render()

        if plotTraining:
            fig = plt.figure()
            ax = fig.add_subplot(111, ylabel='Performance (Sharpe Ratio)', xlabel='Episode')
            ax.plot(performanceTrain)
            ax.plot(performanceTest)
            ax.legend(["Training", "Testing"])
            plt.savefig(''.join(['Figures/', str(marketSymbol), '_TrainingTestingPerformance', '.png']))
            #plt.show()
            for i in range(len(trainingEnvList)):
                self.plotTraining(score[i][:episode], marketSymbol)

        if showPerformance:
            analyser = PerformanceEstimator(trainingEnv.data)
            analyser.displayPerformance('TDQN')
        
        self.writer.close()
        
        return trainingEnv


    def testing(self, trainingEnv, testingEnv, rendering=False, showPerformance=False):
        dataAugmentation = DataAugmentation()
        testingEnvSmoothed = dataAugmentation.lowPassFilter(testingEnv, filterOrder)
        trainingEnv = dataAugmentation.lowPassFilter(trainingEnv, filterOrder)

        # Initialization of some RL variables
        coefficients = self.getNormalizationCoefficients(trainingEnv)
        state = self.processState(testingEnvSmoothed.reset(), coefficients)
        testingEnv.reset()
        QValues0 = []
        QValues1 = []
        done = 0

        # Interact with the environment until the episode termination
        while done == 0:
            action, _, QValues = self.chooseAction(state)
                
            nextState, _, done, _ = testingEnvSmoothed.step(action)
            testingEnv.step(action)
                
            state = self.processState(nextState, coefficients)
            QValues0.append(QValues[0])
            QValues1.append(QValues[1])

        if rendering:
            testingEnv.render()
            self.plotQValues(QValues0, QValues1, testingEnv.marketSymbol)

        if showPerformance:
            analyser = PerformanceEstimator(testingEnv.data)
            analyser.displayPerformance('TDQN')
        
        return testingEnv


    def plotTraining(self, score, marketSymbol):
        fig = plt.figure()
        ax1 = fig.add_subplot(111, ylabel='Total reward collected', xlabel='Episode')
        ax1.plot(score)
        plt.savefig(''.join(['Figures/', str(marketSymbol), 'TrainingResults', '.png']))
        #plt.show()

    
    def plotQValues(self, QValues0, QValues1, marketSymbol):
        fig = plt.figure()
        ax1 = fig.add_subplot(111, ylabel='Q values', xlabel='Time')
        ax1.plot(QValues0)
        ax1.plot(QValues1)
        ax1.legend(['Short', 'Long'])
        plt.savefig(''.join(['Figures/', str(marketSymbol), '_QValues', '.png']))
        #plt.show()


    def plotExpectedPerformance(self, trainingEnv, trainingParameters=[], iterations=10):
        dataAugmentation = DataAugmentation()
        trainingEnvList = dataAugmentation.generate(trainingEnv)

        # Save the initial Deep Neural Network weights
        initialWeights =  copy.deepcopy(self.policyNetwork.state_dict())

        performanceTrain = np.zeros((trainingParameters[0], iterations))
        performanceTest = np.zeros((trainingParameters[0], iterations))

        marketSymbol = trainingEnv.marketSymbol
        startingDate = trainingEnv.endingDate
        endingDate = '2025-1-1'
        money = trainingEnv.data['Money'][0]
        stateLength = trainingEnv.stateLength
        transactionCosts = trainingEnv.transactionCosts
        testingEnv = TradingEnv(marketSymbol, startingDate, endingDate, money, stateLength, transactionCosts)

        print("Hardware selected for training: " + str(self.device))
      
        try:
            for iteration in range(iterations):
                print(''.join(["Expected performance evaluation progression: ", str(iteration+1), "/", str(iterations)]))
                for episode in tqdm(range(trainingParameters[0])):
                    for i in range(len(trainingEnvList)):
                        coefficients = self.getNormalizationCoefficients(trainingEnvList[i])
                        trainingEnvList[i].reset()
                        startingPoint = random.randrange(len(trainingEnvList[i].data.index))
                        trainingEnvList[i].setStartingPoint(startingPoint)
                        state = self.processState(trainingEnvList[i].state, coefficients)
                        previousAction = 0
                        done = 0
                        stepsCounter = 0

                        while done == 0:
                            action, _, _ = self.chooseActionEpsilonGreedy(state, previousAction)
                            nextState, reward, done, info = trainingEnvList[i].step(action)

                            reward = self.processReward(reward)
                            nextState = self.processState(nextState, coefficients)
                            self.replayMemory.push(state, action, reward, nextState, done)

                            otherAction = int(not bool(action))
                            otherReward = self.processReward(info['Reward'])
                            otherDone = info['Done']
                            otherNextState = self.processState(info['State'], coefficients)
                            self.replayMemory.push(state, otherAction, otherReward, otherNextState, otherDone)

                            stepsCounter += 1
                            if stepsCounter == learningUpdatePeriod:
                                self.learning()
                                stepsCounter = 0

                            state = nextState
                            previousAction = action
                
                    # Compute both training and testing  current performances
                    trainingEnv = self.testing(trainingEnv, trainingEnv)
                    analyser = PerformanceEstimator(trainingEnv.data)
                    performanceTrain[episode][iteration] = analyser.computeSharpeRatio()
                    self.writer.add_scalar('Training performance (Sharpe Ratio)', performanceTrain[episode][iteration], episode)     
                    testingEnv = self.testing(trainingEnv, testingEnv)
                    analyser = PerformanceEstimator(testingEnv.data)
                    performanceTest[episode][iteration] = analyser.computeSharpeRatio()
                    self.writer.add_scalar('Testing performance (Sharpe Ratio)', performanceTest[episode][iteration], episode)

                # Restore the initial state of the intelligent RL agent
                if iteration < (iterations-1):
                    trainingEnv.reset()
                    testingEnv.reset()
                    self.policyNetwork.load_state_dict(initialWeights)
                    self.targetNetwork.load_state_dict(initialWeights)
                    self.optimizer = optim.Adam(self.policyNetwork.parameters(), lr=learningRate, weight_decay=L2Factor)
                    self.replayMemory.reset()
                    self.iterations = 0
                    stepsCounter = 0
            
            iteration += 1
        
        except KeyboardInterrupt:
            print()
            print("WARNING: Expected performance evaluation prematurely interrupted...")
            print()
            self.policyNetwork.eval()

        # Compute the expected performance of the intelligent DRL trading agent
        expectedPerformanceTrain = []
        expectedPerformanceTest = []
        stdPerformanceTrain = []
        stdPerformanceTest = []
        for episode in range(trainingParameters[0]):
            expectedPerformanceTrain.append(np.mean(performanceTrain[episode][:iteration]))
            expectedPerformanceTest.append(np.mean(performanceTest[episode][:iteration]))
            stdPerformanceTrain.append(np.std(performanceTrain[episode][:iteration]))
            stdPerformanceTest.append(np.std(performanceTest[episode][:iteration]))
        expectedPerformanceTrain = np.array(expectedPerformanceTrain)
        expectedPerformanceTest = np.array(expectedPerformanceTest)
        stdPerformanceTrain = np.array(stdPerformanceTrain)
        stdPerformanceTest = np.array(stdPerformanceTest)

        # Plot each training/testing iteration performance of the intelligent DRL trading agent
        for i in range(iteration):
            fig = plt.figure()
            ax = fig.add_subplot(111, ylabel='Performance (Sharpe Ratio)', xlabel='Episode')
            ax.plot([performanceTrain[e][i] for e in range(trainingParameters[0])])
            ax.plot([performanceTest[e][i] for e in range(trainingParameters[0])])
            ax.legend(["Training", "Testing"])
            plt.savefig(''.join(['Figures/', str(marketSymbol), '_TrainingTestingPerformance', str(i+1), '.png']))
            #plt.show()

        # Plot the expected performance of the intelligent DRL trading agent
        fig = plt.figure()
        ax = fig.add_subplot(111, ylabel='Performance (Sharpe Ratio)', xlabel='Episode')
        ax.plot(expectedPerformanceTrain)
        ax.plot(expectedPerformanceTest)
        ax.fill_between(range(len(expectedPerformanceTrain)), expectedPerformanceTrain-stdPerformanceTrain, expectedPerformanceTrain+stdPerformanceTrain, alpha=0.25)
        ax.fill_between(range(len(expectedPerformanceTest)), expectedPerformanceTest-stdPerformanceTest, expectedPerformanceTest+stdPerformanceTest, alpha=0.25)
        ax.legend(["Training", "Testing"])
        plt.savefig(''.join(['Figures/', str(marketSymbol), '_TrainingTestingExpectedPerformance', '.png']))
        #plt.show()

        # Closing of the tensorboard writer
        self.writer.close()
        
        return trainingEnv

        
    def saveModel(self, fileName):
        torch.save(self.policyNetwork.state_dict(), fileName)


    def loadModel(self, fileName):
        self.policyNetwork.load_state_dict(torch.load(fileName, map_location=self.device))
        self.targetNetwork.load_state_dict(self.policyNetwork.state_dict())


    def plotEpsilonAnnealing(self):
        plt.figure()
        plt.plot([self.epsilonValue(i) for i in range(10*epsilonDecay)])
        plt.xlabel("Iterations")
        plt.ylabel("Epsilon value")
        plt.savefig(''.join(['Figures/', 'EpsilonAnnealing', '.png']))
        #plt.show()



'''
We took inspiration for the architecture of the Deep Reinforcement Learning (DRL) algorithm from the paper
titled "An application of deep reinforcement learning to algorithmic trading" by Thibaut Théate and Damien Ernst.

We also took help from ChatGPT and Claude to better understand the cocepts and the implementation of the DRL algorithm.
'''
