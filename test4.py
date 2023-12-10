import random
import matplotlib.pyplot as plt

point = []
for i in range(100):
    x = i
    y = random.random()/2
    point.append((x, y))

x = [point[0] for point in point]
y = [point[1] for point in point]
fig, ax = plt.subplots()
ax.plot(x, y)
ax.axis([0,100,0,1])
plt.show()