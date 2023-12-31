import random
import pickle
import math
import numpy as np
import csv
import matplotlib.pyplot as plt

CP_info = []
with open("CP_info.csv", "r") as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        row_int = [int(item) for item in row]
        CP_info.append(row_int)
#CP_info:[0-31+1]*[0-31+1]の2次元リスト

with open('opu_new.pickle', 'rb') as f:
    mapper = pickle.load(f)
    print("test.")

nodes = mapper.default_targets
START_POINT = mapper.starting_point[0]
nodes.append(START_POINT)
#CP:0-31,START:32
e_path = mapper.paths


num_vehicle:int = 2
num_visit:int = 50
num_cities:int = 32

battery_capacity:int = 3000

mu = 0.001279214

population_size:int = 30
generations = 1000
crossover_rate = 0.5
mutation_rate = 0.05

#経路を分かりやすく表示
Route = list[int]
AllRobotRoutes = list[Route]

#初期個体生成
#各個体は1-32のランダム順列(経路長：台数*50)
def generate_individual() -> list[AllRobotRoutes] :
    genes: list[AllRobotRoutes] = []
    #提案手法
    for g in range(population_size):
        one_gene: AllRobotRoutes = []
        for _ in range(num_vehicle):
            one_gene.append([random.randint(0, num_cities-1) for _ in range(num_visit)])
        genes.append(one_gene)
    
    #従来の各CPを一度ずつ通る方法(後で実装)
    
    return genes

#充電の考慮：拠点に戻るタイミング(32)の挿入
def go_depot(route: Route, capacity) -> Route:
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
def rebuilding(genes: list[AllRobotRoutes]) -> list[AllRobotRoutes]:
    battery_capacity = 3000
    
    rebuilded_path: list[AllRobotRoutes] = []
    for all_robot_routes in genes:
        rebuilded_part = []
        for route in all_robot_routes:
            rebuilded_part.append(go_depot(route, battery_capacity))
        rebuilded_path.append(rebuilded_part)        
    return rebuilded_path

    # この書き方のほうがシンプル（分かりにくいかも？）
    # return [
    #     [go_depot(route, battery_capacity) for route in all_robot_routes]
    #     for all_robot_routes in genes
    # ]

def evaluate_fitness(genes: list[AllRobotRoutes]):
    evaluated_data = dict()
    #各個体に関するvisit_list
    for i_gene, gene in enumerate(genes):
        visit_list = [[] for _ in range(num_cities+1)]
        max_time = 0
        for j in range(len(gene)):#個体内の各車両について
            time = 0
            start = num_cities
            for k in range(len(gene[j])):
                goal = genes[i][j][k]
                time += CP_info[start][goal]
                #visit_listを各車両ごとに追加
                visit_list[goal].append(time)
                start = goal
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
            #積の平均にする場合
            #uncertainty = math.pow(math.prod(uncertainty_list), 1/len(uncertainty_list))
            #最悪値にする場合
            #uncertainty = max(uncertainty_list)
        evaluated_data[i_gene] = uncertainty
    #評価順にソート()不確かさの小さい順
    evaluated_sorted = sorted(evaluated_data.items(), key = lambda x :x[1])
    return evaluated_sorted #[(個体番号0-29, 評価値),...]が昇順にソートされたリスト(中身はタプル)


#2点交叉
def crossover(parent1, parent2):
    child1 = [p1[:] for p1 in parent1]
    child2 = [p2[:] for p2 in parent2]
    for i in range(len(parent1)):
        part_rate = random.random()
        if part_rate < crossover_rate:
            gene_length = len(parent1[i])
            crossover_point1 = random.randint(0, gene_length - 1)
            crossover_point2 = random.randint(crossover_point1, gene_length - 1)
            # 2点交叉を実行
            child1[i] = parent1[i][:crossover_point1] + parent2[i][crossover_point1:crossover_point2] + parent1[i][crossover_point2:]
            child2[i] = parent2[i][:crossover_point1] + parent1[i][crossover_point1:crossover_point2] + parent2[i][crossover_point2:]
    return child1, child2

#突然変異
def mutation(parent):
    mutated = parent
    for i in range(len(mutated)):
        for j in range(len(mutated[i])):
            if random.random() < mutation_rate:
                mutated[i][j] = random.randint(0,num_cities)
    return mutated

#エリート保存
def elite_selection(population, fitness, index):
    elite = population[fitness[index][0]]
    return elite

#経路の可視化
def show_route(path):
    color_list = ["r","b","g","y","m","c","k","w"]
    fig = plt.figure()
    route_index = [] #経路を2重リストから1重リストへ
    for i in range(len(path)):
        route_index.extend(path[i])
        route_index.pop()
    
    for i in range(len(path)):
        route = []
        for j in range(len(path[i])-1):
        #CP番号→座標
            start = nodes[path[i][j]]
            goal = nodes[path[i][j+1]]
        #移動ルートをrouteに追加
            if start != goal:
                temp_route = e_path[(start, goal)][0]
                route.extend(temp_route)
                route.pop()
            #データをxとyに分割
        x = [point[0] for point in route]
        y = [point[1] for point in route]
    # 散布図をプロット
        plt.scatter(x, y, s=1, marker='.', c=color_list[i])
    
    plt.title('Surveilance Route')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    # グリッドを表示
    plt.grid(True)

    # グラフを表示
    plt.show()
    return

#ここから実行本体
print('Start.')
now_genes: list[AllRobotRoutes] = generate_individual()

best = 10000
a = 1
fitness = []
while a <= generations:
    path = rebuilding(now_genes) #デコーディング
    genes_fitness = evaluate_fitness(path) #適応度の計算
    #最良評価の個体を出力
    # genes_fitness の最初のタプルから最良個体のインデックスを取得
    best_individual_index = genes_fitness[0][0]
    # 取得したインデックスを使って now_genes から最良個体にアクセス
    best_individual = now_genes[best_individual_index]
    best_path = path[best_individual_index]
    best_fitness = genes_fitness[0][1]
    temp_fitness = (a, best_fitness)
    fitness.append(temp_fitness)


    #評価値を更新したら表示
    if best_fitness < best:
        best = best_fitness
        best_solution = best_path
        print("Gen.", a,':', best_fitness)
    
    #世代更新
    next_genes = []
    #エリート保存
    next_genes.append(best_individual)
    childs = 0

    while childs < population_size:
        parents_num = random.choices(genes_fitness[0], k=2, weights = genes_fitness[1])
        num1 = int(parents_num[0])
        num2 = int(parents_num[1])
        #2点交叉
        child1, child2 = crossover(now_genes[num1], now_genes[num2])
        #突然変異
        child_mutated1 = mutation(child1)
        child_mutated2 = mutation(child2)

        next_genes.append(child1)
        next_genes.append(child2)

        childs+=2
    #next_genesをnow_genesに置き換え
    del next_genes[-1]
    now_genes = next_genes
    a += 1
#適応度推移のグラフ
x0 = [point[0] for point in fitness]
y0 = [point[1] for point in fitness]
plt.scatter(x0, y0, s=1)
plt.title('Fitness')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()

#最良な解を出力する
print(best_solution)
show_route(best_solution)
print('End.')    




