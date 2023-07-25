import random
import numpy
import math

mu = 0.001279214
t = 1000

p = 1 - math.exp(-mu * t)
print(p)

u = ((t) + ((1/mu) * math.exp(-mu*t)) - 1/mu)/t
print(math.exp(-mu*t))
print(u)

'''
def list_test(n):
    child1 = [n,3,4]
    child2 = [n,6,7]
    return child1, child2

test_list = []

child1, child2 = list_test(100)
test_list.append(child1)
test_list.append(child2)
print(test_list)



def generate_individual(vehicle, visit, population, cities):
    genes = []
    for g in range(population):
        one_gene = []
        for h in range(vehicle):
            one_gene.append([random.randint(0,cities-1)for i in range(visit)])
        genes.append(one_gene)
    return genes
now_genes = generate_individual(3,5,4,32)
print(now_genes)

evaluated_data = {0:0.8, 1:0.6, 2:0.78, 3:0.66}
evaluated_sorted = sorted(evaluated_data.items(), key = lambda x :x[1])
print(evaluated_sorted)

def elite_selection(population, fitness, index):
    elite = population[fitness[index][0]]
    return elite

childs = 0
next_genes = []
while childs < 2:
    next_genes.append(elite_selection(now_genes, evaluated_sorted, childs)) #ceil:切り上げ
    childs+=1

print(next_genes)


print(8-2-3)

import csv

CP_info = []
with open("CP_info.csv", "r") as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        row_int = [int(item) for item in row]
        CP_info.append(row_int)
print(CP_info)
'''