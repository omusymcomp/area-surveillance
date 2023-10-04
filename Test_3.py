#正規分布(ガウス分布)

import random
import matplotlib.pyplot as plt

seed = 13
random.seed(seed)

#size
height = 300
width = 300

# 生成する点の数
num_cities = 32  # 任意の数を設定

# 正規分布に従った点を生成し、指定の範囲にスケーリングして整数に変換
scaled_points = []

for _ in range(num_cities):
    x = int(random.gauss(0.5, 0.2) * height)  # 平均0.5、標準偏差0.2の正規分布に従った乱数生成、スケーリング、整数化
    y = int(random.gauss(0.5, 0.2) * width)  # 同様にY座標を生成、スケーリング、整数化
    scaled_points.append((x, y))

# 生成した整数座標を表示
for point in scaled_points:
    print(f"X: {point[0]}, Y: {point[1]}")

x = [point[0] for point in scaled_points]
y = [point[1] for point in scaled_points]
fig = plt.figure()
plt.scatter(x, y)
plt.grid(True)
plt.show()