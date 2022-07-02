import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SA():
    def __init__(self,L,k,first_temperature,last_temperature,city_num,distance_matrix):
        self.L=L
        self.k=k
        self.first_temperature=first_temperature
        self.last_temperature=last_temperature
        self.current_temperature=self.first_temperature
        self.fitness=1e4
        self.city_num=city_num
        self.individual=np.random.permutation(self.city_num)
        self.distance_matrix=distance_matrix
        self.history_fitness=[]

    def fitness_calculation(self):
        temp=0
        for i in range(self.city_num-1):
            start=self.individual[i]
            end=self.individual[i+1]
            temp+=self.distance_matrix[start,end]
        start=self.individual[0]
        end=self.individual[-1]
        temp+=self.distance_matrix[end,start]
        self.fitness=temp

    def temp_fitness_calculation(self,individual_temp):
        temp=0
        for i in range(self.city_num-1):
            start=individual_temp[i]
            end=individual_temp[i+1]
            temp+=self.distance_matrix[start,end]
        start=individual_temp[0]
        end=individual_temp[-1]
        temp+=self.distance_matrix[end,start]
        return temp


    def isothermal_process(self):
        for i in range(self.L):
            temp1=np.random.randint(self.city_num)
            temp2=np.random.randint(self.city_num)
            start=min(temp1,temp2)
            end=max(temp1,temp2)
            #print('start',start)
            #print('end',end)
            individual_temp=self.individual.copy()
            #print('individual_temp',individual_temp)
            #print('individual_temp[end-1:start-1:-1]',individual_temp[end-1:start-1:-1])
            #print('individual_temp[start:end]',individual_temp[start:end])
            if end!=0:#避免0为末端索引引起的倒序空集
                if start!=0:#避免0为起始索引使倒序产生空集
                    individual_temp[start:end]=individual_temp[end-1:start-1:-1]
                else:
                    individual_temp[start:end] = individual_temp[end - 1::-1]


            temp_fitness=self.temp_fitness_calculation(individual_temp)
            self.fitness_calculation()
            if self.fitness>temp_fitness:
                self.individual=individual_temp
            else:
                if (np.e)**(-(temp_fitness-self.fitness)/self.current_temperature)>np.random.rand():
                    self.individual=individual_temp

    def temperature_fall_process(self):
        while self.current_temperature>self.last_temperature:
            self.isothermal_process()
            self.current_temperature*=self.k
            self.history_fitness.append(self.fitness)




data=pd.read_excel('data_generation.xlsx')
x=data['x']
y=data['y']

distance_matrix=np.zeros((len(x),len(y)))
for i in range(len(x)):
    for j in range(len(y)):
        distance_matrix[i, j] = ((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2) ** 0.5

test=SA(L=1000,k=0.99,first_temperature=100,last_temperature=0.1,city_num=30,distance_matrix=distance_matrix)
test.temperature_fall_process()

plt.figure()
for i in range(len(test.individual)-1):
    start=test.individual[i]
    end=test.individual[i+1]
    plt.scatter([x[start],x[end]],[y[start],y[end]])
    plt.plot([x[start],x[end]],[y[start],y[end]])
start=test.individual[0];end=test.individual[-1]
plt.scatter([x[start],x[end]],[y[start],y[end]])
plt.plot([x[start],x[end]],[y[start],y[end]])
plt.show()

plt.figure()
plt.plot(test.history_fitness)
plt.show()

print('最短路径',test.individual)
print('最短距离',test.fitness)

