import random
import math
import matplotlib.pyplot as plt

#フィールド設定
height = 600 #縦
width = 600 #横
num_cities = 32

seed = 1

#ランダムに都市を生成する
def generate_cities(height, width, num_cities, seed):
    random.seed(seed)
    nodes = [-1] * num_cities
    for i in range(num_cities):
        x = random.randint(0,width)
        y = random.randint(0,height)
        nodes[i] = (x, y)
    return nodes

nodes = generate_cities(height, width, num_cities, seed)

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
print("Created.")

fig = plt.figure()

#データをxとyに分割
x = [point[0] for point in nodes]
y = [point[1] for point in nodes]
# depox = nodes[len(nodes)-1][0]
# depoy = nodes[len(nodes)-1][1]

# 散布図をプロット
plt.scatter(x, y)
# plt.scatter(depox,depoy,c='r')

# グラフにタイトルと軸ラベルを追加
plt.title('Scatter Plot of Coordinates')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# グリッドを表示
plt.grid(True)

# グラフを表示
plt.show()
