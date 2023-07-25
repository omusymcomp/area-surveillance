import random
import math
import numpy as np
import csv

CP_info = []
with open("CP_info.csv", "r") as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        row_int = [int(item) for item in row]
        CP_info.append(row_int)
#CP_info:[0-31+1]*[0-31+1]の2次元リスト

num_vehicle:int = 3
num_visit:int = 50
num_cities:int = 32

battery_capacity:int = 3000

mu = 0.001279214

population_size:int = 30
generations = 1000
crossover_rate = 0.5
crossover_rate2 = 0.5
mutation_rate = 0.9
mutation_rate2 = 0.05

#初期個体生成
#各個体は1-32のランダム順列(経路長：台数*50)
def generate_individual(vehicle, visit, population, cities):
    genes = []
    for g in range(population):
        one_gene = []
        for h in range(vehicle):
            one_gene.append([random.randint(0,cities-1)for i in range(visit)])
        genes.append(one_gene)
    return genes

#充電の考慮：拠点に戻るタイミング(32)の挿入
def go_depot(route, capacity):
    new_route = [num_cities] #拠点からスタート
    temp = num_cities
    now_battery = capacity
    for i in route:
        if now_battery - CP_info[temp][i] - CP_info[i][num_cities]>=0: #充電足りる
            new_route.append(i)
            now_battery -= CP_info[temp][i]
            temp = i
        else:
            new_route.append(num_cities)
            now_battery = capacity
            temp = num_cities
            new_route.append(i)
            now_battery -= CP_info[temp][i]
    new_route.append(num_cities)#拠点でゴール
    return new_route

#genesからrouteへ変換
#1.各個体ごとに分割, 2.各車両ごとに分割, 3.充電ごとに分割
def rebuilding(path_origin):
    battery_capacity = 3000
    rebuilded_path = []
    for i in range(len(path_origin)):
        rebuilded_part = []
        for j in range(len(path_origin[i])):
            rebuilded_part.append(go_depot(path_origin[i][j],battery_capacity))
        rebuilded_path.append(rebuilded_part)        
    return rebuilded_path

#経路長の計算(不使用)
#path:経路
def sum_path(path):
    length = 0
    return length

#genesにはrebuilded_pathを入れる
def evaluate_fitness(genes):
    evaluated_data = dict()
    #各個体に関するvisit_list
    for i in range(len(genes)):
        visit_list = [[] for _ in range(num_cities+1)]
        max_time = 0
        for j in range(len(genes[i])):#個体内の各車両について
            time = 0
            s = num_cities
            for k in range(len(genes[i][j])):
                g = genes[i][j][k]
                time += CP_info[s][g]
                #visit_listを各車両ごとに追加
                visit_list[g].append(time)
                s = g
            if time > max_time: #車両に割り当てられた経路長の最大を記憶
                max_time = time
        uncertainty_list = []
        for l in range(len(visit_list)):
            if not visit_list[l]:
                uncertainty_list.append(10000)
            else:
                visit_list[l].sort()
                #各CPごとに不確かさを計算
                now = 0
                u = 0
                for m in range(len(visit_list[l])):
                    u += (visit_list[l][m]-now) + (1/mu) * math.exp(-mu*(visit_list[l][m]-now)) - 1/mu
                    #visit_list[l][m]:時間
                    now = visit_list[l][m]
                u += (max_time-now) + (1/mu) * math.exp(-mu*(max_time-now)) - 1/mu
                uncertainty_list.append(u/max_time)
            #不確かさの平均
            uncertainty = np.mean(uncertainty_list)
        evaluated_data[i] = uncertainty
    #評価順にソート()不確かさの小さい順
    evaluated_sorted = sorted(evaluated_data.items(), key = lambda x :x[1])
    return evaluated_sorted #[(個体番号0-29, 評価値),...]が昇順にソートされたリスト(中身はタプル)

#エリート選択
def elite_selection(population, fitness, index):
    elite = population[fitness[index][0]]
    return elite

#2点交叉
def crossover(parent1, parent2):
    child1 = [p1[:] for p1 in parent1]
    child2 = [p2[:] for p2 in parent2]
    for i in range(len(parent1)):
        part_rate = random.random()
        if part_rate < crossover_rate2:
            gene_length = len(parent1[i])
            crossover_point1 = random.randint(0, gene_length - 1)
            crossover_point2 = random.randint(crossover_point1, gene_length - 1)
            # 2点交叉を実行
            child1[i] = parent1[i][:crossover_point1] + parent2[i][crossover_point1:crossover_point2] + parent1[i][crossover_point2:]
            child2[i] = parent2[i][:crossover_point1] + parent1[i][crossover_point1:crossover_point2] + parent2[i][crossover_point2:]
    return child1, child2

#突然変異
def mutation(parent, bonus):
    mutated = parent
    for i in range(len(mutated)):
        for j in range(len(mutated[i])):
            if random.random() < mutation_rate2 + bonus:
                mutated[i][j] = random.randint(0,num_cities)
    return mutated

#ここから実行本体
print('Start.')
now_genes = generate_individual(num_vehicle, num_visit, population_size, num_cities)

best = 10000
mutation_bonus = 0
a = 1
while a <= generations:
    path = rebuilding(now_genes) #デコーディング
    genes_fitness = evaluate_fitness(path) #評価値の計算
    #最良評価の個体を出力
    # genes_fitness の最初のタプルから最良個体のインデックスを取得
    best_individual_index = genes_fitness[0][0]
    # 取得したインデックスを使って now_genes から最良個体にアクセス
    best_individual = now_genes[best_individual_index]
    best_path = path[best_individual_index]
    best_fitness = genes_fitness[0][1]
    
    #評価値を更新したら表示
    if best_fitness < best:
        mutation_bonus = 0
        best = best_fitness
        best_solution = best_path
        print("Gen.", a,':', best_fitness)
    #世代更新(選択、交叉、突然変異)
    next_genes = []
    childs = 0
    #選択(全体の1割から切り上げて偶数へ)
    while childs < math.ceil(population_size/20) * 2:
        next_genes.append(elite_selection(now_genes, genes_fitness, childs)) #ceil:切り上げ
        childs+=1

    while childs < population_size:
        parent1_num = random.randint(0,population_size-1)
        parent2_num = random.randint(0,population_size-1)
        r = random.random()
        #交叉
        if r <= crossover_rate:
            child1, child2 = crossover(now_genes[parent1_num], now_genes[parent2_num])
            next_genes.append(child1)
            next_genes.append(child2)
        #突然変異
        elif r <= mutation_rate:
            child_mutated1 = mutation(now_genes[parent1_num], mutation_bonus)
            next_genes.append(child_mutated1)
            child_mutated2 = mutation(now_genes[parent1_num], mutation_bonus)
            next_genes.append(child_mutated2)

        #コピー
        else:
            next_genes.append(now_genes[parent1_num])
            next_genes.append(now_genes[parent2_num])
        childs += 2
    #next_genesをnow_genesに置き換え
    now_genes = next_genes
    mutation_bonus += 1/generations #世代更新されないほど変異率を上げる
    a += 1

#print(generate_individual(num_vehicle, num_visit, population_size, num_cities))
print(best_solution) 
print('End.')