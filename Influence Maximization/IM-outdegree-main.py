import sys
import numpy as np
import datetime
from math import ceil,exp
from random import randint,choice,randrange
import random
import argparse
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import copy
#import pickle5 as pickle
class ObjectiveIM(object):
    def __init__(self, weightMatrix,nodeNum,outdegree_file,budget):
        self.weightMatrix = weightMatrix
        self.n=nodeNum
        self.budget=budget
        self.solution = []
        self.allNodes=np.ones((1, self.n))
        self.cost=[0]*self.n

        dataFile=open(outdegree_file)
        dataLine=dataFile.readlines()
        items=dataLine[0].split()
        eps=[0]*self.n
        for i in range(self.n):
            eps[i]=float(items[i])
        dataFile.close()
       
        for i in range(self.n):
            outDegree=(self.weightMatrix[i,:]>0).sum()
            self.cost[i]=1.0+(1+abs(eps[i]))*outDegree

    def setBudget(self,budget):
        self.budget=budget

    def setEvaluateTime(self,time):
        self.evaluateTime=time
    
    def setDynamic_evaluateTime(self,time):
        self.dynamic_evaluateTime=time

    def FinalActiveNodes(self):  # solution is the numpy matrix 1*n
        activeNodes = np.zeros((1, self.n)) + self.solution
        cActive = np.zeros((1, self.n)) + self.solution# currently active nodes
        tempNum = int(cActive.sum(axis=1)[0, 0])
        while tempNum > 0:
            nActive = self.allNodes - activeNodes
            randMatirx = np.random.rand(tempNum, self.n)#uniformly random matrix between 0 and 1
            z = sum(randMatirx < self.weightMatrix[cActive.nonzero()[-1], :]) > 0 #cActive.nonzero()[-1] is the nonzero index
            cActive = np.multiply(nActive, z) #sum is the sum of each column,the new active node
            activeNodes = (cActive + activeNodes) > 0
            tempNum = int(cActive.sum(axis=1)[0, 0])
        return activeNodes.sum(axis=1)[0, 0]

    def Position(self, s):
        return np.array(np.where(s[0, :] == 1)[1]).reshape(-1)

    def FS(self, solution):  # simulate 500 times
        self.solution = solution
        val = 0
        for i in range(500):
            val += self.FinalActiveNodes()
        return val / 500.0

    def CS(self,s):
        tempSum=0
        pos=self.Position(s)
        for item in pos:
            tempSum=tempSum+self.cost[item]
        return tempSum

    def save_result(self,res_file,times,tempmax1,cost,budget,result):
        log = open(res_file+'/result_'+str(times)+'.txt', 'a')
        log.write("value = "+str(tempmax1) +" cost = "+str(cost)+" budget = "+str(budget))
        log.write("\n")
        for item in self.Position(result):
            log.write(str(item))
            log.write(" ")
        log.write("\n")
        log.close()

    def AGGA(self,new_budget):
        if(new_budget<self.budget):
            while(self.CS(self.result)>new_budget):
                c = self.CS( self.result)
                f = self.FS( self.result)
                minVolume=float('inf')
                for j in range(0 ,self.n):
                    if( self.result[0,j]==1):
                        self.result[0,j]=0
                        fv = self.FS( self.result)
                        cv = self.CS( self.result)
                        tempVolume = 1.0 * (f - fv) / (c - cv)
                        if tempVolume<minVolume:
                            minVolume=tempVolume
                            selectedIndex=j
                        self.result[0,j]=1
                self.result[0,selectedIndex]=0

        elif(new_budget>self.budget):
            #V'=V \ X
            V_pi = [1] * self.n
            for i in range(0 ,self.n):
                if(self.result[0,i]==1):
                    V_pi[i]=0
            selectedIndex=0

            while sum(V_pi) > 0:
                # print(sum(V_pi))
                f = self.FS(self.result)
                c = self.CS(self.result)
                maxVolume = -1
                for j in range(0, self.n):
                    if V_pi[j] == 1:
                        self.result[0, j] = 1
                        cv = self.CS(self.result)
                        if cv > self.budget:
                            self.result[0, j] = 0
                            V_pi[j] = 0
                            continue
                        fv = self.FS(self.result)
                        tempVolume = 1.0 * (fv - f) / (cv - c)
                        if tempVolume > maxVolume:
                            maxVolume = tempVolume
                            selectedIndex = j
                        self.result[0, j] = 0
                self.result[0, selectedIndex] = 1
                if self.CS(self.result) > self.budget:
                    self.result[0, selectedIndex] = 0
                V_pi[selectedIndex] = 0

        tempMax = 0
        tempresult = np.mat(np.zeros((1, self.n)), 'int8')
        selectedSingleton=0
        for i in range(self.n):
            if self.cost[i] <= new_budget:
                tempresult[0, i] = 1
                tempVolume = self.FS(tempresult)
                if tempVolume > tempMax:
                    tempMax = tempVolume
                    selectedSingleton=i
                tempresult[0, i] = 0
        tempresult[0,selectedSingleton]=1

        tempmax1 = self.FS(self.result)
        if tempmax1 < tempMax:
            tempmax1=tempMax
            self.result=tempresult

        self.setBudget(new_budget)
        
        return tempmax1,self.CS(self.result),self.budget,self.result     

    def GGA(self):
        self.result = np.mat(np.zeros((1, self.n)), 'int8')
        V_pi = [1] * self.n
        selectedIndex = 0
        #summ=0
        while sum(V_pi) > 0:
            # print(sum(V_pi))
            f = self.FS(self.result)
            c = self.CS(self.result)
            maxVolume = -1
            for j in range(0, self.n):
                if V_pi[j] == 1:
                    self.result[0, j] = 1
                    cv = self.CS(self.result)
                    if cv > self.budget:
                        self.result[0, j] = 0
                        V_pi[j] = 0
                        continue
                    fv = self.FS(self.result)
                    tempVolume = 1.0 * (fv - f) / (cv - c)
                    if tempVolume > maxVolume:
                        maxVolume = tempVolume
                        selectedIndex = j
                    self.result[0, j] = 0
            self.result[0, selectedIndex] = 1
            # if self.CS(self.result) > self.budget:
            #     self.result[0, selectedIndex] = 0
            V_pi[selectedIndex] = 0

        tempMax = 0.
        tempresult = np.mat(np.zeros((1, self.n)), 'int8')
        selectedSingleton=0
        for i in range(self.n):
            if self.cost[i] <= self.budget:
                tempresult[0, i] = 1
                tempVolume = self.FS(tempresult)
                if tempVolume > tempMax:
                    tempMax = tempVolume
                    selectedSingleton=i
                tempresult[0, i] = 0
        tempresult[0,selectedSingleton]=1

        tempmax1 = self.FS(self.result)
        if tempmax1 < tempMax:
            tempmax1=tempMax
            self.result=tempresult

        return tempmax1,self.CS(self.result),self.budget,self.result  
 
    def mutation(self, s):
        rand_rate = 1.0 / (self.n)  
        change = np.random.binomial(1, rand_rate, self.n)
        return np.abs(s - change)
    
    def POMC(self,res_file,times):
        population = np.mat(np.zeros([1, self.n], 'int8'))  # initiate the population
        fitness = np.mat(np.zeros([1, 2]))
        popSize = 1
        iter = 0
        nn = int(self.n * self.n)
        with tqdm(range(int(self.evaluateTime*self.n* self.n)), position = 0, leave = True) as pbar:# progress bar which represents the total number of evaluations
            for _ in pbar:
                if iter == nn:
                    iter = 0
                    resultIndex = -1
                    maxValue = float("-inf")
                    for p in range(0, popSize):
                        if fitness[p, 1] <= self.budget and fitness[p, 0] > maxValue:
                            maxValue = fitness[p, 0]
                            resultIndex = p
                    # tempValue=self.EstimateObjective_accurate(population[resultIndex,:])
                    log = open(res_file+'/result_'+str(times)+'.txt', 'a')
                    log.write(" value = "+str(fitness[resultIndex,0])+" cost = "+str(fitness[resultIndex,1])+ " budget = " + str(self.budget) + " population = "+str(popSize))
                    log.write("\n")
                    for item in self.Position(population[resultIndex,:]):
                        log.write(str(item))
                        log.write(" ")
                    log.write("\n")
                    log.close()
            
                iter += 1
                s = population[randint(1, popSize) - 1, :]  # choose a individual from population randomly
                offSpring = self.mutation(s)  # every bit will be flipped with probability 1/n
                offSpringFit = np.mat(np.zeros([1, 2]))  # comparable value, size, original value
                offSpringFit[0, 1] = self.CS(offSpring)
                if offSpringFit[0, 1] == 0 or offSpringFit[0, 1] > self.budget+1:
                    continue
                offSpringFit[0, 0] = self.FS(offSpring)
                hasBetter = False
                for i in range(0, popSize):
                    if (fitness[i, 0] > offSpringFit[0, 0] and fitness[i, 1] <= offSpringFit[0, 1]) or (
                            fitness[i, 0] >= offSpringFit[0, 0] and fitness[i, 1] < offSpringFit[0, 1]):
                        hasBetter = True
                        break
                if hasBetter == False:  # there is no better individual than offSpring
                    Q = []
                    for j in range(0, popSize):
                        if offSpringFit[0, 0] >= fitness[j, 0] and offSpringFit[0, 1] <= fitness[j, 1]:
                            continue
                        else:
                            Q.append(j)
                    # Q.sort()
                    fitness = np.vstack((offSpringFit, fitness[Q, :]))  # update fitness
                    population = np.vstack((offSpring, population[Q, :]))  # update population
               
                popSize = np.shape(fitness)[0]

    def setInilSolution(self):
        self.population = np.mat(np.zeros([1, self.n], 'int8'))
        self.fitness = np.mat(np.zeros([1, 2]))  
        # update the value of function f and c
        self.fitness[0, 1] = self.CS(self.population[0, :])
        self.fitness[0, 0] = self.FS(self.population[0, :])

    def DY_POMC(self,res_file,times):
        popSize = np.shape(self.population)[0]

        with tqdm(range(int(self.dynamic_evaluateTime)), position = 0, leave = True) as pbar:# progress bar which represents the total number of evaluations
            for _ in pbar:             
                s = self.population[randint(1, popSize) - 1, :]  # choose a individual from population randomly
                offSpring = self.mutation(s) # every bit will be flipped with probability 1/n
                
                offSpringFit = np.mat(np.zeros([1, 2]))  # comparable value, size, original value
                offSpringFit[0, 1] = self.CS(offSpring)
                
                if offSpringFit[0, 1] == 0 or offSpringFit[0, 1] > self.budget+1:
                    continue

                offSpringFit[0, 0] = self.FS(offSpring)

                hasBetter = False
                for i in range(0, popSize):
                    if (self.fitness[i, 0] > offSpringFit[0, 0] and self.fitness[i, 1] <= offSpringFit[0, 1]) or (
                            self.fitness[i, 0] >= offSpringFit[0, 0] and self.fitness[i, 1] < offSpringFit[0, 1]):
                        hasBetter = True
                        break
                if hasBetter == False:  # there is no better individual than offSpring
                    Q = []
                    for j in range(0, popSize):
                        if offSpringFit[0, 0] >= self.fitness[j, 0] and offSpringFit[0, 1] <= self.fitness[j, 1]:
                            continue
                        else:
                            Q.append(j)
                    # Q.sort()
                  
                    self.fitness = np.vstack((offSpringFit, self.fitness[Q, :]))  # update fitness
                    self.population = np.vstack((offSpring, self.population[Q, :]))  # update population
                popSize = np.shape(self.fitness)[0]
    
        resultIndex = -1
        maxValue = float("-inf")
        for p in range(0, popSize):
            if self.fitness[p, 1] <= self.budget and self.fitness[p, 0] > maxValue:
                maxValue = self.fitness[p, 0]
                resultIndex = p
        
        log = open(res_file+'/result_'+str(times)+'.txt', 'a')
        log.write(" value = "+str(self.fitness[resultIndex,0])+" cost = "+str(self.fitness[resultIndex,1])+ " budget = " + str(self.budget) + " population = "+str(popSize))
        log.write("\n")
        for item in self.Position(self.population[resultIndex,:]):
            log.write(str(item))
            log.write(" ")
        log.write("\n")
        log.close()  

    def GS(self, alpha, offSpringFit):
        if offSpringFit[0, 2] >= 1:
            return 1.0 * offSpringFit[0, 0] / (1.0 - (1.0 / exp(alpha * offSpringFit[0, 1] / self.budget)))
        else:
            return 0

    def EAMC(self,res_file,times):  ##just consider cost is less B
        X = np.mat(np.zeros([self.n + 1, self.n], 'int8'))  # initiate the population
        Y = np.mat(np.zeros([self.n + 1, self.n], 'int8'))  # initiate the population
   
        population = np.mat(np.zeros([1, self.n], 'int8'))
        Xfitness = np.mat(np.zeros([self.n + 1, 4]))  # f(s), c(s),|s|,g(s)
        Yfitness = np.mat(np.zeros([self.n + 1, 4]))  # f(s), c(s),|s|,g(s)

        offSpringFit = np.mat(np.zeros([1, 4]))  # f(s),c(s),|s|,g(s)
        xysame = [0] * (self.n + 1)
        zwsame = [0] * (self.n + 1)
        xysame[0] = 1
        zwsame[0] = 1

        popSize = 1
        t = 0  # the current iterate count
        iter1 = 0
        nn = int(0.05*self.n * self.n)
        with tqdm(range(int(self.evaluateTime*self.n* self.n)), position = 0, leave = True) as pbar:# progress bar which represents the total number of evaluations
            for _ in pbar:
                if iter1 == nn:
                    iter1 = 0
                    YresultIndex = -1
                    maxValue = float("-inf")
                    for p in range(0, self.n + 1):
                        if Yfitness[p, 1] <= self.budget and Yfitness[p, 0] > maxValue:
                            maxValue = Yfitness[p, 0]
                            YresultIndex = p

                    XresultIndex = -1
                    maxValue = float("-inf")
                    for p in range(0, self.n + 1):
                        if Xfitness[p, 1] <= self.budget and Xfitness[p, 0] > maxValue:
                            maxValue = Xfitness[p, 0]
                            XresultIndex = p
                    
                    log = open(res_file+'/result_'+str(times)+'.txt', 'a')
                    if Yfitness[YresultIndex, 0] > Xfitness[XresultIndex, 0]:
                        log.write(" value = "+str(Yfitness[YresultIndex,0])+" cost = "+str(Yfitness[YresultIndex,1])+" population = "+str(popSize))
                        log.write("\n")
                        for item in self.Position(Y[YresultIndex,:]):
                            log.write(str(item))
                            log.write(" ")
                        log.write("\n")
                        log.close()
                    else:
                        log.write(" value = "+str(Xfitness[XresultIndex,0])+" cost = "+str(Xfitness[XresultIndex,1])+" population = "+str(popSize))
                        log.write("\n")
                        for item in self.Position(X[XresultIndex,:]):
                            log.write(str(item))
                            log.write(" ")
                        log.write("\n")
                        log.close()


                iter1 += 1
                s = population[randint(1, popSize) - 1, :]  # choose a individual from population randomly
                offSpring = self.mutation(s)  # every bit will be flipped with probability 1/n
                offSpringFit[0, 1] = self.CS(offSpring)
                offSpringFit[0, 0] = self.FS(offSpring)
                offSpringFit[0, 2] = offSpring[0, :].sum()
                offSpringFit[0, 3] = self.GS( 1.0, offSpringFit)
                indice = int(offSpringFit[0, 2])
                if offSpringFit[0, 2] < 1:
                    t = t + 1
                    continue
                isadd1 = 0
                isadd2 = 0
                if offSpringFit[0, 1] <= self.budget:
                    if offSpringFit[0, 3] >= Xfitness[indice, 3]:
                        X[indice, :] = offSpring
                        Xfitness[indice, :] = offSpringFit
                        isadd1 = 1
                    if offSpringFit[0, 0] >= Yfitness[indice, 0]:
                        Y[indice, :] = offSpring
                        Yfitness[indice, :] = offSpringFit
                        isadd2 = 1
                    if isadd1 + isadd2 == 2:
                        xysame[indice] = 1
                    else:
                        if isadd1 + isadd2 == 1:
                            xysame[indice] = 0
                # count the population size
                tempSize = 1  # 0^n is always in population
                for i in range(1, self.n + 1):
                    if Xfitness[i, 2] > 0:
                        if Yfitness[i, 2] > 0 and xysame[i] == 1:  # np.linalg.norm(X[i,:]-Y[i,:])==0: #same
                            tempSize = tempSize + 1
                        if Yfitness[i, 2] > 0 and xysame[i] == 0:  # np.linalg.norm(X[i,:]-Y[i,:])>0:
                            tempSize = tempSize + 2
                        if Yfitness[i, 2] == 0:
                            tempSize = tempSize + 1
                    else:
                        if Yfitness[i, 2] > 0:
                            tempSize = tempSize + 1
                if popSize != tempSize:
                    population = np.mat(np.zeros([tempSize, self.n], 'int8'))
                popSize = tempSize
                j = 1
                # merge the X,Y,
                for i in range(1, self.n + 1):
                    if Xfitness[i, 2] > 0:
                        if Yfitness[i, 2] > 0 and xysame[i] == 1:
                            population[j, :] = X[i, :]
                            j = j + 1
                        if Yfitness[i, 2] > 0 and xysame[i] == 0:
                            population[j, :] = X[i, :]
                            j = j + 1
                            population[j, :] = Y[i, :]
                            j = j + 1
                        if Yfitness[i, 2] == 0:
                            population[j, :] = X[i, :]
                            j = j + 1
                    else:
                        if Yfitness[i, 2] > 0:
                            population[j, :] = Y[i, :]
                            j = j + 1
                t = t + 1
        
    def setInilSolution_EAMC(self):
        self.X = np.mat(np.zeros([self.n + 1, self.n], 'int8'))  # initiate the population  （n+1,n）
        self.Y = np.mat(np.zeros([self.n + 1, self.n], 'int8'))  # initiate the population  （n+1,n）
        self.population = np.mat(np.zeros([1, self.n], 'int8'))  #（1,n）
        
        self.Xfitness = np.mat(np.zeros([self.n + 1, 4]))  # f(s), c(s),|s|,g(s) （n+1,4）
        self.Yfitness = np.mat(np.zeros([self.n + 1, 4]))  # f(s), c(s),|s|,g(s) （n+1,4）
      
        self.xysame = [0] * (self.n + 1)
        self.xysame[0] = 1
          
    def reCalculate(self):
        self.population = np.mat(np.zeros([1, self.n], 'int8'))
        zero_solution=np.mat(np.zeros([1, self.n], 'int8'))
        for p in range(1, self.n + 1):
            if self.Yfitness[p, 2] > 0:
                #when the budget changes, the infeasible solution in the current popualtion will be deleted
                if self.Yfitness[p, 1] > self.budget: 
                    self.Y[p,:]=np.zeros([1, self.n])
                    self.Yfitness[p, 0] = self.FS(self.Y[p,:])
                    self.Yfitness[p, 1] = self.CS(self.Y[p,:])
                    self.Yfitness[p, 2] = self.Yfitness[p, :].sum()
                    self.Yfitness[p, 3] = self.GS( 1.0, self.Y[p,:])
                else:
                    self.Yfitness[p, 0] = self.FS(self.Y[p,:])
                    self.Yfitness[p, 3] = self.GS( 1.0, self.Y[p,:])

            if self.Xfitness[p, 2] > 0:
                #when the budget changes, the infeasible solution in the current popualtion will be deleted
                if self.Xfitness[p, 1] > self.budget:
                    self.X[p,:]=np.zeros([1, self.n])
                    self.Xfitness[p, 0] = self.FS(self.Y[p,:])
                    self.Xfitness[p, 1] = self.CS(self.Y[p,:])
                    self.Xfitness[p, 2] = self.Xfitness[p, :].sum()
                    self.Xfitness[p, 3] = self.GS( 1.0, self.Y[p,:])
                else:
                    self.Xfitness[p, 0] = self.FS(self.Y[p,:])
                    self.Xfitness[p, 3] = self.GS( 1.0, self.Y[p,:])


            if np.array_equal(self.X[p, 0], self.Y[p, 0]):
                if not np.array_equal(self.Y[p, 0], zero_solution):
                    self.population = np.vstack((self.Y[p, :], self.population))  
            else:
                if not np.array_equal(self.Y[p, 0], zero_solution):
                    self.population = np.vstack((self.Y[p, :], self.population))  
                if not np.array_equal(self.X[p, 0], zero_solution):
                    self.population = np.vstack((self.X[p, :], self.population)) 

    def DY_EAMC(self,res_file,times):  ##just consider cost is less B
        popSize = np.shape(self.population)[0]

        with tqdm(range(int(self.dynamic_evaluateTime)), position = 0, leave = True) as pbar:# progress bar which represents the total number of evaluations
            for _ in pbar:
                index=randint(1, popSize) - 1
                s = self.population[index, :]  # choose a individual from population randomly
                offSpring = self.mutation(s)  # every bit will be flipped with probability 1/n
                offSpringFit = np.mat(np.zeros([1, 4])) # f(s),c(s),|s|,g(s)
                offSpringFit[0, 1] = self.CS(offSpring)
                offSpringFit[0, 0] = self.FS(offSpring)
                offSpringFit[0, 2] = offSpring[0, :].sum()
                offSpringFit[0, 3] = self.GS( 1.0, offSpringFit)
               

                if offSpringFit[0, 2] < 1:
                    continue
                
                isadd1 = 0
                isadd2 = 0
                if offSpringFit[0, 1] <= self.budget:
                    indice = int(offSpringFit[0, 2]) # bin

                    if offSpringFit[0, 3] >= self.Xfitness[indice, 3]: #g(x')\ge g(u^i)
                        self.X[indice, :] = offSpring
                        self.Xfitness[indice, :] = offSpringFit
                        isadd1 = 1
                    if offSpringFit[0, 0] >= self.Yfitness[indice, 0]: #f(x')\ge f(u^i)
                        self.Y[indice, :] = offSpring
                        self.Yfitness[indice, :] = offSpringFit
                        isadd2 = 1
                    if isadd1 + isadd2 == 2:
                        self.xysame[indice] = 1
                    elif isadd1 + isadd2 == 1:
                        self.xysame[indice] = 0

                # count the population size
                tempSize = 1  # 0^n is always in population
                for i in range(1, self.n + 1):
                    if self.Xfitness[i, 2] > 0: # |x_i| > 0
                        if self.Yfitness[i, 2] > 0 and self.xysame[i] == 1:  
                            tempSize = tempSize + 1
                        if self.Yfitness[i, 2] > 0 and self.xysame[i] == 0:  
                            tempSize = tempSize + 2
                        if self.Yfitness[i, 2] == 0:
                            tempSize = tempSize + 1
                    else:
                        if self.Yfitness[i, 2] > 0:
                            tempSize = tempSize + 1
               
                if popSize != tempSize:
                    self.population = np.mat(np.zeros([tempSize, self.n], 'int8'))
                popSize = tempSize
                
                j = 1
                # merge the X,Y
                for i in range(1, self.n + 1):
                    if self.Xfitness[i, 2] > 0:
                        if self.Yfitness[i, 2] > 0 and self.xysame[i] == 1:
                            self.population[j, :] = self.X[i, :]
                            j = j + 1
                        if self.Yfitness[i, 2] > 0 and self.xysame[i] == 0:
                            self.population[j, :] = self.X[i, :]
                            j = j + 1
                            self.population[j, :] = self.Y[i, :]
                            j = j + 1
                        if self.Yfitness[i, 2] == 0:
                            self.population[j, :] = self.X[i, :]
                            j = j + 1
                    else:
                        if self.Yfitness[i, 2] > 0:
                            self.population[j, :] = self.Y[i, :]
                            j = j + 1

           
        YresultIndex = -1
        maxValue = float("-inf")
        for p in range(0, self.n + 1):
            if self.Yfitness[p, 1] <= self.budget and self.Yfitness[p, 0] > maxValue:
                maxValue = self.Yfitness[p, 0]
                YresultIndex = p

        XresultIndex = -1
        maxValue = float("-inf")
        for p in range(0, self.n + 1):
            if self.Xfitness[p, 1] <= self.budget and self.Xfitness[p, 0] > maxValue:
                maxValue = self.Xfitness[p, 0]
                XresultIndex = p
        
        log = open(res_file+'/result_'+str(times)+'.txt', 'a')
        if self.Yfitness[YresultIndex, 0] > self.Xfitness[XresultIndex, 0]:
            log.write(" value = "+str(self.Yfitness[YresultIndex,0])+" cost = "+str(self.Yfitness[YresultIndex,1])+" population = "+str(popSize))
            log.write("\n")
            for item in self.Position(self.Y[YresultIndex,:]):
                log.write(str(item))
                log.write(" ")
            log.write("\n")
            log.close()
        else:
            log.write(" value = "+str(self.Xfitness[XresultIndex,0])+" cost = "+str(self.Xfitness[XresultIndex,1])+" population = "+str(popSize))
            log.write("\n")
            for item in self.Position(self.X[XresultIndex,:]):
                log.write(str(item))
                log.write(" ")
            log.write("\n")
            log.close()

    def h(self,z, x_f,x_c):
        C = 100000
        if x_c > z['c_value']:
            return (x_f - z['f_value']) / ( x_c - z['c_value'])
        else:
            return (x_f - z['f_value']) * C + z['c_value'] -  x_c
        
    def select(self):
        # Filter out empty sets
        non_empty_P=[]
        p_vs_fitness=[]
        for index, p in enumerate(self.P):
            if np.shape(p)[0] > 0:
                non_empty_P.append(p)
                p_vs_fitness.append(self.fitness[index])

        # non_empty_P = [p for p in self.P if np.shape(p)[0] > 0]  

        i = randint(0, len(non_empty_P)-1)
        P_i = non_empty_P[i]

        #Find the nearest reference point less than i
        list=[k for k in range(i) if self.refer_points[k]]
        #If the selected subpopulation is i=0, there is no reference point, and the returned solution is naturally zero.
        if not list:
            return P_i[randint(0, np.shape(P_i)[0] - 1):,]
        
        k = max(list)
        z = self.refer_points[k] 

        # Finding the S set
        # Initialize the maximum h value and the corresponding solution set
        max_h_value = float('-inf')
        S = []

        for index, x in enumerate(P_i):
            # Calculate the h value of the current solution
            current_h_value = self.h(z,p_vs_fitness[i][index,0], p_vs_fitness[i][index,1])
            
            # Check whether the h value of the current solution is the new maximum value
            if current_h_value > max_h_value:
                # Update the maximum h value
                max_h_value = current_h_value
                # Reset the solution set because a new maximum h value has been found
                S = [x]
            elif current_h_value == max_h_value:
                # If the h value of the current solution is the same as the known maximum h value, it is added to the solution set
                S.append(x)

        if 'point' in self.refer_points[i]:
            flag=False
            for matrix2 in S:
                if np.array_equal(self.refer_points[i]['point'], matrix2):
                    flag=True
            if flag:
                s = self.refer_points[i]['point'] 

            else: 
                s=S[randint(0, len(S)-1)]

        else: 
            s=S[randint(0, len(S)-1)]

        if np.random.rand() < 0.5:
            x = s 
        else:
            x=P_i[randint(0, np.shape(P_i)[0]-1),:]
     

        return x

    def local_search(self,x):
        sum=0
        y = copy.deepcopy(x['point'])
        y_f = self.FS(y)
        y_c = self.CS(y)
        sum += 1
        for i in range(self.n):
            if x['point'][0, i] == 0:
                s = copy.deepcopy(x['point'])
                s[0, i] = 1
                s_c=self.CS(s)
                if s_c <= self.budget:
                    s_f = self.FS(s)
                    sum += 1
                    if self.h(x, s_f,s_c) >= self.h(x, y_f, y_c):
                        y = s
                        y_f = s_f 
                        y_c = s_c
                        
        return y, y_f, y_c, sum

    def setInilSolution_FPOMC(self):
        subP_0 = np.mat(np.zeros((1, self.n), dtype='int8'))
        self.P = [[] for _ in range(self.n+1)]
        self.P[0]=subP_0

        subFitness_0=np.mat(np.zeros([1, 2]))
        self.fitness = [[] for _ in range(self.n+1)]
        self.fitness[0]= subFitness_0

        self.refer_points = [[] for _ in range(self.n+1)]
        self.refer_points[0]={'point':np.mat(np.zeros((1, self.n)), dtype='int8'),'f_value':0,'c_value':0}

    def DY_FPOMC(self,res_file,times):
        # We set the predetermined end value of the progress bar
        desired_progress = int(self.dynamic_evaluateTime)

        pbar = tqdm()

        current_progress = 0

        while current_progress < desired_progress:
            x = self.select()
            x_prime = self.mutation(x)  # every bit will be flipped with probability 1/n
            
            offSpringFit = np.mat(np.zeros([1, 2])) 
            offSpringFit[0, 1] = self.CS(x_prime)
            if offSpringFit[0, 1] == 0 or offSpringFit[0, 1] > self.budget+1:
                continue
            offSpringFit[0, 0] = self.FS(x_prime)
            
            j = np.sum(x_prime)
            popSize_j = np.shape(self.P[j])[0]
            hasBetter = False
            for i in range(0, popSize_j):
                if (self.fitness[j][i, 0] > offSpringFit[0, 0] and self.fitness[j][i, 1] <= offSpringFit[0, 1]) or (
                        self.fitness[j][i, 0] >= offSpringFit[0, 0] and self.fitness[j][i, 1] < offSpringFit[0, 1]):
                    hasBetter = True
                    break

            if hasBetter == False:  # there is no better individual than offSpring
                Q = []
                for q in range(0, popSize_j):
                    if offSpringFit[0, 0] >= self.fitness[j][q, 0] and offSpringFit[0, 1] <= self.fitness[j][q, 1]:
                        continue
                    else:
                        Q.append(q)
          

                if np.shape(self.fitness[j])[0]==0:
                    self.fitness[j] = offSpringFit
                    self.P[j] = x_prime
                else:
                    self.fitness[j] = np.vstack((offSpringFit, self.fitness[j][Q,:]))  # update fitness
                    self.P[j] = np.vstack((x_prime, self.P[j][Q, :]))  # update population

                if not self.refer_points[j]:
                    self.refer_points[j]={'point':x_prime,'f_value': offSpringFit[0, 0],'c_value': offSpringFit[0, 1]}
                else:
                    k = max([k for k in range(j) if self.refer_points[k]])
                    z = self.refer_points[k]
                    value=self.h(z, offSpringFit[0, 0], offSpringFit[0, 1])
                    if value >= self.h(z, self.refer_points[j]['f_value'], self.refer_points[j]['c_value']):
                        _,ff,cc,tt=self.local_search(z)
                        # Record the number of evaluations used in LS
                        progress_increment = tt
                        current_progress += progress_increment
                        # Update progress bar
                        pbar.update(progress_increment)

                        if value >= self.h(z,ff,cc):
                            self.refer_points[j]={'point':x_prime,'f_value': offSpringFit[0, 0],'c_value': offSpringFit[0, 1]}
                            y,ff,cc,tt = self.local_search(self.refer_points[j])
                             # Record the number of evaluations used in LS
                            progress_increment = tt
                            current_progress += progress_increment
                            # Update progress bar
                            pbar.update(progress_increment)

                            offSpringFit = np.mat(np.zeros([1, 2]))  # comparable value, size, original value
                            offSpringFit[0, 1] = cc 

                            if offSpringFit[0, 1] <= self.budget:
                                offSpringFit[0, 0] = ff
                                popSize_j_plus_1 = np.shape(self.P[j+1])[0] 

                                hasBetter = False
                                for i in range(0, popSize_j_plus_1):
                                    if (self.fitness[j+1][i, 0] > offSpringFit[0, 0] and self.fitness[j+1][i, 1] <= offSpringFit[0, 1]) or (
                                            self.fitness[j+1][i, 0] >= offSpringFit[0, 0] and self.fitness[j+1][i, 1] < offSpringFit[0, 1]):
                                        hasBetter = True
                                        break
                                if hasBetter == False:  # there is no better individual than offSpring
                                    Q = []
                                    for q in range(0, popSize_j_plus_1):
                                        if offSpringFit[0, 0] >= self.fitness[j+1][q, 0] and offSpringFit[0, 1] <= self.fitness[j+1][q, 1]:
                                            continue
                                        else:
                                            Q.append(q)

                                    if np.shape(self.fitness[j+1])[0]==0:
                                        self.fitness[j+1] = offSpringFit
                                        self.P[j+1] = y
                                    else:
                                        self.fitness[j+1] = np.vstack((offSpringFit, self.fitness[j+1][Q,:]))  # update fitness
                                        self.P[j+1] = np.vstack((y, self.P[j+1][Q, :]))  # update population
           
            progress_increment = 1
            current_progress += progress_increment
         
            pbar.update(progress_increment)

            if current_progress >= desired_progress:
                break
        pbar.close()
        
        resultIndex = -1
        subpopulation = -1
        maxValue = float("-inf")
        popSize_sum=0
        for P_index in range(self.n+1):
            popSize=np.shape(self.P[P_index])[0]
            popSize_sum+=popSize
            for p in range(0, popSize):
                if self.fitness[P_index][p, 1] <= self.budget and self.fitness[P_index][p, 0] > maxValue:
                    maxValue = self.fitness[P_index][p, 0]
                    resultIndex = p
                    subpopulation = P_index
                    
        log = open(res_file+'/result_'+str(times)+'.txt', 'a')
        log.write(" value = "+str(self.fitness[subpopulation][resultIndex,0])+" cost = "+str(self.fitness[subpopulation][resultIndex,1])+ " budget = " + str(self.budget) + " population = "+str(popSize_sum))
        log.write("\n")
        for item in self.Position(self.P[subpopulation][resultIndex,:]):
            log.write(str(item))
            log.write(" ")
        log.write("\n")
        log.close()          
     
    def select_weight(self):#biased selection
        values = np.array([matrix[0, -1] for matrix in self.fitness])

        epsilon = 1e-10
        weights = 1 / (np.abs(values - self.budget) + epsilon)
        weights /= weights.sum()

        chosen_index = np.random.choice(len(values), p=weights)
        return self.population[chosen_index, :] 
        
    def DY_BPODC_cold(self,res_file,times):
        popSize = np.shape(self.population)[0]

        with tqdm(range(int(self.dynamic_evaluateTime)), position = 0, leave = True) as pbar:# progress bar which represents the total number of evaluations
            for _ in pbar:             
                s = self.select_weight()
                offSpring = self.mutation(s)  # every bit will be flipped with probability 1/n
                offSpringFit = np.mat(np.zeros([1, 2]))  # comparable value, size, original value
                offSpringFit[0, 1] = self.CS(offSpring)
                
                if offSpringFit[0, 1] == 0 or offSpringFit[0, 1] > self.budget+1:
                    continue

                offSpringFit[0, 0] = self.FS(offSpring)

                hasBetter = False
                for i in range(0, popSize):
                    if (self.fitness[i, 0] > offSpringFit[0, 0] and self.fitness[i, 1] <= offSpringFit[0, 1]) or (
                            self.fitness[i, 0] >= offSpringFit[0, 0] and self.fitness[i, 1] < offSpringFit[0, 1]):
                        hasBetter = True
                        break
                    
                if hasBetter == False:  # there is no better individual than offSpring
                    Q = []
                    for j in range(0, popSize):
                        if offSpringFit[0, 0] >= self.fitness[j, 0] and offSpringFit[0, 1] <= self.fitness[j, 1]:
                            continue
                        else:
                            Q.append(j)
                  
                    self.fitness = np.vstack((offSpringFit, self.fitness[Q, :]))  # update fitness
                    self.population = np.vstack((offSpring, self.population[Q, :]))  # update population
                popSize = np.shape(self.fitness)[0]
    
        resultIndex = -1
        maxValue = float("-inf")
        for p in range(0, popSize):
            if self.fitness[p, 1] <= self.budget and self.fitness[p, 0] > maxValue:
                maxValue = self.fitness[p, 0]
                resultIndex = p
        
        log = open(res_file+'/result_'+str(times)+'.txt', 'a')
        log.write(" value = "+str(self.fitness[resultIndex,0])+" cost = "+str(self.fitness[resultIndex,1])+ " budget = " + str(self.budget) + " population = "+str(popSize))
        log.write("\n")
        for item in self.Position(self.population[resultIndex,:]):
            log.write(str(item))
            log.write(" ")
        log.write("\n")
        log.close() 

    

    def setWarm(self,flag):
        self.warmUp=flag

    def DY_BPODC(self,res_file,times):
        popSize = np.shape(self.population)[0]

        with tqdm(range(int(self.dynamic_evaluateTime)), position = 0, leave = True) as pbar:# progress bar which represents the total number of evaluations
            for _ in pbar:   
                if self.warmUp:
                    parent = self.population[randint(1, popSize) - 1, :]
                else:
                    parent = self.select_weight()
                offSpring = self.mutation(parent)  # every bit will be flipped with probability 1/n
                offSpringFit = np.mat(np.zeros([1, 2]))  # comparable value, size, original value
                offSpringFit[0, 1] = self.CS(offSpring)
                
                if offSpringFit[0, 1] == 0 or offSpringFit[0, 1] > self.budget+1:
                    continue

                offSpringFit[0, 0] = self.FS(offSpring)

                hasBetter = False
                for i in range(0, popSize):
                    if (self.fitness[i, 0] > offSpringFit[0, 0] and self.fitness[i, 1] <= offSpringFit[0, 1]) or (
                            self.fitness[i, 0] >= offSpringFit[0, 0] and self.fitness[i, 1] < offSpringFit[0, 1]):
                        hasBetter = True
                        break
                    
                if hasBetter == False:  # there is no better individual than offSpring
                    Q = []
                    for j in range(0, popSize):
                        if offSpringFit[0, 0] >= self.fitness[j, 0] and offSpringFit[0, 1] <= self.fitness[j, 1]:
                            continue
                        else:
                            Q.append(j)
                  
                    self.fitness = np.vstack((offSpringFit, self.fitness[Q, :]))  # update fitness
                    self.population = np.vstack((offSpring, self.population[Q, :]))  # update population
                popSize = np.shape(self.fitness)[0]
    
        resultIndex = -1
        maxValue = float("-inf")
        for p in range(0, popSize):
            if self.fitness[p, 1] <= self.budget and self.fitness[p, 0] > maxValue:
                maxValue = self.fitness[p, 0]
                resultIndex = p
        
        log = open(res_file+'/result_'+str(times)+'.txt', 'a')
        log.write(" value = "+str(self.fitness[resultIndex,0])+" cost = "+str(self.fitness[resultIndex,1])+ " budget = " + str(self.budget) + " population = "+str(popSize))
        log.write("\n")
        for item in self.Position(self.population[resultIndex,:]):
            log.write(str(item))
            log.write(" ")
        log.write("\n")
        log.close() 
             
def ReadData(p,filePath):
    dataFile=open(filePath)
    maxNode=0
    while True:
        line=dataFile.readline()
        if not line:
            break
        items=line.split()
        if len(items)>0:
            start=int(items[0])
            end=int(items[1])
            if start>maxNode:
                maxNode=start
            if end>maxNode:
                maxNode=end
    dataFile.close()
    maxNode=maxNode

    data = np.mat(np.zeros([maxNode, maxNode]))
    dataFile = open(filePath)
    while True:
        line = dataFile.readline()
        if not line:
            break
        items = line.split()
        if len(items)>0:
            data[int(items[0])-1,int(items[1])-1]=p
    dataFile.close()
    return data

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def main(args):
    # save args settings
    adjacency_file="outdegree/"+args.adjacency_file
    outdegree_file="outdegree/"+args.outdegree_file
    algo=args.algo
    probability=args.probability
    budget=args.budget
    T=args.T
    times=args.times
    crossover_prob=args.crossover_prob
    print(adjacency_file,budget,algo, T)

    #problem
    weightMatrix=ReadData(probability,adjacency_file)
    nodeNum=np.shape(weightMatrix)[0]
    myObject=ObjectiveIM(weightMatrix,nodeNum,outdegree_file,budget)


    if(args.ifDynamic):
        #get dynamic budget
        dynamic_budget_file=args.dynamic_budget_file
        dataFile=open(dynamic_budget_file)
        dataLine=dataFile.readlines()
        items=dataLine[0].split()
        dynamic_budget=[0]*100
        dynamic_budget[0]=budget+float(items[0])
        for i in range(1,100):
            dynamic_budget[i]=dynamic_budget[i-1]+float(items[i])
            dynamic_budget[i]=max(args.budget_min,dynamic_budget[i])
            dynamic_budget[i]=min(args.budget_max,dynamic_budget[i])
        dataFile.close()


        #dynamic result file

        res_file='dynamic/'+ args.adjacency_file + '_' + dynamic_budget_file + '_'+ algo
        if not os.path.exists(res_file):
            os.makedirs(res_file)
       

        warmUp_evaluateTime=int(nodeNum*(nodeNum+1)/2/4)
        dynamic_evaluateTime=int(nodeNum*(nodeNum+1)/2/2)

        #run dynamic algo
        if algo=="GGA":
            #original budget
            print("original budget 0 :"+ str(budget)+ "...")
            tempmax1,cost,old_budget,result=myObject.GGA()
            myObject.save_result(res_file,times,tempmax1,cost,old_budget,result)
            #dynamic budget
            for i in range(len(dynamic_budget)):
                print("dynamic budget "+ str(i) + ":" + str(dynamic_budget[i])+ "...")
                myObject.setBudget(dynamic_budget[i])
                tempmax1,cost,old_budget,result=myObject.GGA()
                myObject.save_result(res_file,times,tempmax1,cost,old_budget,result)
        
        elif algo=="AGGA":
            #original budget
            tempmax1,cost,old_budget,result=myObject.GGA()
            myObject.save_result(res_file,times,tempmax1,cost,old_budget,result)
            #dynamic budget
            for i in range(len(dynamic_budget)):
                tempmax1,cost,old_budget,result=myObject.AGGA(dynamic_budget[i])
                myObject.save_result(res_file,times,tempmax1,cost,old_budget,result)

        elif algo=="DY_POMC":
            myObject.setInilSolution()  #zero solution as the initial solution

            #warm-up for the initial budget
            myObject.setDynamic_evaluateTime(warmUp_evaluateTime)
            myObject.DY_POMC(res_file,times)
            
            myObject.setDynamic_evaluateTime(dynamic_evaluateTime)
            for i in range(len(dynamic_budget)):
                myObject.setBudget(dynamic_budget[i])
                myObject.DY_POMC(res_file,times)

        elif algo=="DY_EAMC":
            myObject.setInilSolution_EAMC()

            myObject.setDynamic_evaluateTime(warmUp_evaluateTime)
            myObject.DY_EAMC(res_file,times)

            myObject.setDynamic_evaluateTime(dynamic_evaluateTime)
            for i in range(len(dynamic_budget)):
                myObject.setBudget(dynamic_budget[i])
                myObject.reCalculate()
                myObject.DY_EAMC(res_file,times)

        elif algo=="DY_FPOMC":
            myObject.setInilSolution_FPOMC()

            myObject.setDynamic_evaluateTime(warmUp_evaluateTime)
            myObject.DY_FPOMC(res_file,times)

            myObject.setDynamic_evaluateTime(dynamic_evaluateTime)
            for i in range(len(dynamic_budget)):
                myObject.setBudget(dynamic_budget[i])
                myObject.DY_FPOMC(res_file,times)

        elif algo=="DY_BPODC_cold":
            myObject.setInilSolution()  #zero solution as the initial solution

            myObject.setDynamic_evaluateTime(warmUp_evaluateTime)
            myObject.DY_BPODC_cold(res_file,times)
            
            myObject.setDynamic_evaluateTime(dynamic_evaluateTime)
            for i in range(len(dynamic_budget)):
                myObject.setBudget(dynamic_budget[i])
                myObject.DY_BPODC_cold(res_file,times) 

        elif algo=="DY_BPODC":
            myObject.setInilSolution()  #zero solution as the initial solution

            myObject.setDynamic_evaluateTime(warmUp_evaluateTime)
            myObject.setWarm(True)
            myObject.DY_BPODC(res_file,times)
            
            myObject.setDynamic_evaluateTime(dynamic_evaluateTime)
            myObject.setWarm(False)
            for i in range(len(dynamic_budget)):
                myObject.setBudget(dynamic_budget[i])
                myObject.DY_BPODC(res_file,times) 

    
        else:
            print("no suitable algo")
    
    else:
        res_file='static_result/'+ args.adjacency_file+ '_'+ algo +'_' + str(budget) 
        if not os.path.exists(res_file):
            os.makedirs(res_file)

        if algo=="GGA":
            myObject.GGA(res_file)
        elif algo=="POMC":
            myObject.setEvaluateTime(T)      
            myObject.POMC(res_file,times)
        elif algo=="EAMC":
            myObject.setEvaluateTime(T)      
            myObject.EAMC(res_file,times)
        else:
            print("no suitable algo")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('-adjacency_file',type=str,default= "graph100-01.txt")
    argparser.add_argument('-outdegree_file',type=str,default= "graph100_eps.txt")
    argparser.add_argument('-budget',type=float, default=300)
    argparser.add_argument('-probability',type=float, default=0.05)
    argparser.add_argument('-algo',type=str, default="DY_BPODC")
    argparser.add_argument('-times', type=int, default=0)
    argparser.add_argument('-T', type=int, default=20)

    argparser.add_argument('-ifDynamic',type=str2bool, default=True)
    argparser.add_argument('-dynamic_budget_file',type=str,default= "dynamic_budget_IM.txt")
    argparser.add_argument('-budget_min',type=float, default=100)
    argparser.add_argument('-budget_max',type=float, default=500)
    args = argparser.parse_args()
    main(args)
