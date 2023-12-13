import random
import pickle
import math
import numpy as np
import csv
import matplotlib.pyplot as plt

#従来手法

#作成したマップで回す場合
global_seed = random.random()

# #フィールド設定
height = 300 #縦
width = 300 #横
num_cities = 32
depot = (150, 150)


#ランダムに都市を生成する(一様分布)
def generate_cities(height, width, num_cities):
    nodes = [-1] * (num_cities + 1)
    random.seed(114)
    for i in range(num_cities):
        x = random.randint(0,width)
        y = random.randint(0,height)
        nodes[i] = (x, y)
    nodes[num_cities] = depot
    return nodes

nodes = generate_cities(height, width, num_cities)

#都市間の経路長を計算
def Chebyshev_distance(nodes):
    CP_info = []
    for i in range(len(nodes)):
        temp_row = []
        for j in range(len(nodes)):
            temp_row.append(max(abs(nodes[i][0]-nodes[j][0]), abs(nodes[i][1]-nodes[j][1]))) #2CP間のチェビシェフ距離
        CP_info.append(temp_row)
    return CP_info

CP_info = Chebyshev_distance(nodes)

#経路を1マスずつ
def create_routes(nodes):
    e_path = {}
    for start in nodes:
        for goal in nodes:
            if start != goal:
                deltaX = goal[0] - start[0]
                deltaY = goal[1] - start[1]
                delta_dif = abs(abs(deltaY) - abs(deltaX)) #斜め移動しない分
                path = []
                temp = [start[0], start[1]] #タプルは更新不可なので、リストで扱う
                while int(temp[0]) != goal[0] and int(temp[1]) != goal[1]: #斜め移動
                    path.append((int(temp[0]), int(temp[1])))
                    temp[0] += deltaX/abs(deltaX)
                    temp[1] += deltaY/abs(deltaY)
                
                if int(temp[0]) == goal[0]:
                    path.append((int(temp[0]), int(temp[1])))
                    for i in range(delta_dif):
                        temp[1] += deltaY/abs(deltaY)
                        path.append((int(temp[0]), int(temp[1])))
                
                elif int(temp[1]) == goal[1]:
                    path.append((int(temp[0]), int(temp[1])))
                    for i in range(delta_dif):
                        temp[0] += deltaX/abs(deltaX)
                        path.append((int(temp[0]), int(temp[1])))
                e_path[(start, goal)] = path, len(path)
    return e_path

e_path = create_routes(nodes)
# print("Created.")

# #中百舌鳥キャンパスで回す場合
# CP_info = []
# with open("CP_info.csv", "r") as file:
#     csv_reader = csv.reader(file)
#     for row in csv_reader:
#         row_int = [int(item) for item in row]
#         CP_info.append(row_int)
# #CP_info:[0-31+1]*[0-31+1]の2次元リスト

# with open('opu_new.pickle', 'rb') as f:
#     mapper = pickle.load(f)
#     print("test.")

# nodes = mapper.default_targets
# START_POINT = mapper.starting_point[0]
# nodes.append(START_POINT)
# #CP:0-31,START:32
# e_path = mapper.paths

trials = 2

num_vehicle:int = 2
num_visit:int = 30
num_cities:int = len(nodes) - 1

battery_capacity:int = 3000

mu = 0.001279214

population_size:int = 30
generations = 1000000
crossover_rate = 0.5
mutation_rate = 0.05

#初期個体生成
#各個体は1-32の並び替え順列(経路長：台数*CP数)
def generate_individual(vehicle, visit, population, cities):
    genes = []
    random.seed(global_seed)
    for g in range(population):
        one_gene = []
        for h in range(vehicle):
            one_robot = []
            for i in range(cities):
                one_robot.append(i)
            random.shuffle(one_robot)
            one_gene.append(one_robot)
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
        #積の平均にする場合
        #uncertainty = math.pow(math.prod(uncertainty_list), 1/len(uncertainty_list))
        #最悪値にする場合
        #uncertainty = max(uncertainty_list)
        evaluated_data[i] = uncertainty, max_time
    #評価順にソート()不確かさの小さい順
    evaluated_sorted = sorted(evaluated_data.items(), key = lambda x :x[1])
    evaluated_converted = []

    for item in evaluated_sorted:
        if isinstance(item[1], tuple):
            #ネストされたタプルの要素を展開して新しいタプルを作成し、リストに追加
            new_tuple = (item[0], item[1][0], item[1][1])
            evaluated_converted.append(new_tuple)
        else:
            evaluated_converted.append(item)

    return evaluated_converted #[(個体番号0-29, 評価値),...]が昇順にソートされたリスト(中身はタプル)

def crossover(parent1, parent2):
    #先に経路一つずつに分解
    child1 = []
    child2 = []
    if random.random() < crossover_rate:
        for i in range(len(parent1)):
            parent1_part = parent1[i]
            parent2_part = parent2[i]
            #交叉点をランダムに選ぶ
            cross_point = random.randint(1, len(parent1_part) - 1)
            #子供の遺伝子を初期化
            child1_part = [-1] * len(parent1_part)
            child2_part = [-1] * len(parent2_part)
            # 交叉点より前の部分をコピー
            for j in range(cross_point):
                child1_part[j] = parent1_part[j]
                child2_part[j] = parent2_part[j]
            #交叉点以降の部分を埋める
            for j in range(cross_point, len(parent1_part)):
                gene1 = parent1_part[j]
                gene2 = parent2_part[j]
                #重複をチェックして解消
                while gene1 in child1_part:
                    gene1 = parent2_part[j]
                while gene2 in child2_part:
                    gene2 = parent1_part[j]
                
                child1_part[j] = gene1
                child2_part[j] = gene2
            child1.append(child1_part)
            child2.append(child2_part)                
    else:
        child1 = parent1
        child2 = parent2
    return child1, child2

def mutation(genes):
    mutated_genes = []
    for h in range(len(genes)):
        genes_part = genes[h]
        mutated_genes_part = genes_part
        for i in range(len(genes_part)):
            if random.random() <= mutation_rate:
                mutation_position1 = random.randint(0,31)
                mutation_position2 = random.randint(0,31)
                temp = mutated_genes_part[mutation_position1]
                mutated_genes_part[mutation_position1] = mutated_genes_part[mutation_position2]
                mutated_genes_part[mutation_position2] = temp
        mutated_genes.append(mutated_genes_part)
    return mutated_genes

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
    #グリッドを表示
    plt.grid(True)

    #グラフを保存
    routefile_path = 'out/route' + str(num_try) + '.png'
    fig.savefig(routefile_path, dpi=300)
    #グラフを表示
    #plt.show()
    return

#ここから実行本体
print('Start.')
for num_try in range(1, trials+1, 1):
    print('Simulation' + str(num_try))

    global_seed = random.random()
    random.seed(global_seed)
    now_genes = generate_individual(num_vehicle, num_visit, population_size, num_cities)

    best = 10000
    a = 1
    fitness = []

    #解の更新を保存
    update_history = []
    log = []

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
        best_length = genes_fitness[0][2]


        #評価値を更新したら表示
        if best_fitness < best:
            best = best_fitness
            best_solution = best_path
            update_history.append([a, best_fitness, best_length])
            print("Gen.", a,':', best_fitness, best_length)
        temp_fitness_and_length = (a, best)
        fitness.append(temp_fitness_and_length)

        #最終世代を表示
        if a == generations:
            update_history.append([a, best_fitness, best_length])
            print("Gen.", a, ':', best_fitness, best_length)
        
        #世代更新
        next_genes = []
        #エリート保存
        next_genes.append(best_individual)
        childs = 0

        #個体番号のリストを作成
        individual_numbers = [item[0] for item in genes_fitness]

        while childs < population_size:
            #不確実度を選択確率として使用し、ランダムに2つの個体番号を選択
            parents_num = random.choices(individual_numbers, k=2, weights=[1/item[1] for item in genes_fitness])
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
    #各世代のログをcsv出力
    logfile_path = 'logs/log' + str(num_try) + '.csv'
    logfile = open(logfile_path, mode='w', newline='')
    writer = csv.writer(logfile)
    try:
        writer.writerows(log)
    finally:
        logfile.close()

    #更新をcsvに出力
    outfile_path = 'out/output' + str(num_try) + '.csv'
    file = open(outfile_path, mode='w', newline='')
    writer_2 = csv.writer(file)
    try:
        writer_2.writerows(update_history)
        writer_2.writerows(best_solution)
    finally:
        file.close()

#適応度推移のグラフ
x0 = [point[0] for point in fitness]
y0 = [point[1] for point in fitness]
fig, ax = plt.subplots()
plt.title('Fitness')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
ax.plot(x0, y0)
ax.axis([0,1000000,0,1])
graph_path = 'out/graph' + str(num_try) + '.png'
fig.savefig(graph_path, dpi=300)
#plt.show()


#最良な解を出力する
print(best_solution)
show_route(best_solution)
print('End.')