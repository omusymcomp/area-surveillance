#やりたいこと：各世代の不確実度と経路長を持って来て平均を出す

#ファイルを読み込む
import matplotlib.pyplot as plt
import csv

gen = 100000
trials = 20
uncertainty = [[] for i in range(gen)]
length = [[] for i in range(gen)]
for i in range(trials):

    logfile_name = 'logs/log' + str(i+1) + '.csv'
    with open(logfile_name, "r") as csv_file:
        f = csv.reader(csv_file, delimiter=',')
        count = 0
        for row in f:
            uncertainty[count].append(float(row[1]))
            length[count].append(int(row[2]))
            count += 1

#適応度推移のグラフ(全体の平均)
x = []
y = []
for j in range(gen):
    x.append(j)
    y.append(sum(uncertainty[j])/trials)
fig, ax = plt.subplots()
plt.title('fitness')
plt.xlabel('generation')
plt.ylabel('uncertainty')
ax.plot(x, y)
fig.savefig('logs/log_all.png', dpi=300)

print('End.')


