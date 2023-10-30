import random

num_vehicle = 2
num_visit = 50
num_cities = 32
num_population = 30
num = 300

# def generate_individual(vehicle, visit, population, cities):
#     genes = []
#     for g in range(population):
#         one_gene = []
#         for h in range(vehicle):
#             one_gene.append([random.randint(0,cities-1)for i in range(visit)])
#         genes.append(one_gene)
#     return genes

def generate_individual(vehicle, visit, population, cities):
    genes = []
    random.seed()
    for g in range(population):
        one_gene = []
        for h in range(vehicle):
            one_robot = []
            for i in range(cities):
                one_robot.append(i)
            one_robot.extend([random.randint(0,cities-1)for j in range(visit - cities)])
            random.shuffle(one_robot)
            one_gene.append(one_robot)
        genes.append(one_gene)
    return genes

now_gene = generate_individual(num_vehicle,num_visit,num_population,num_cities)
print(now_gene)