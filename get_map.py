import pickle
import csv
import matplotlib.pyplot as plt

with open('opu_new.pickle', 'rb') as f:
    mapper = pickle.load(f)
    print("test.")

nodes = mapper.default_targets
START_POINT = mapper.starting_point[0]
nodes.append(START_POINT)
e_path = mapper.paths

#CPの組みとその間の移動ルートがセットになった奴：辞書


fig = plt.figure()

#データをxとyに分割
x = [point[0] for point in nodes]
y = [point[1] for point in nodes]

# 都市の通る順番で線を描画
#for i in range(len(nodes) - 1):
    #plt.plot([nodes[i][0], nodes[i + 1][0]], [nodes[i][1], nodes[i + 1][1]], 'b-')

# 散布図をプロット
plt.scatter(x, y)

# グラフにタイトルと軸ラベルを追加
plt.title('Scatter Plot of Coordinates')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# グリッドを表示
plt.grid(True)

# グラフを表示
plt.show()


