import pickle #pickleデータを使う
import csv #csvデータを使う

with open('opu_new.pickle', 'rb') as f:
    mapper = pickle.load(f) #距離のデータが入っているファイルを参照
    #print("test.")
e_path = mapper.paths #e_path:pathsデータを移してきたやつ、各CP間の経路1ピクセルずつ
nodes = mapper.default_targets #mapperのなかのdefault_targets(各CPの座標)を読み込んでいる
START_POINT = mapper.starting_point[0] #ドローンの拠点

print('Test.')
nodes.append(START_POINT)

count_a = 0
count_b = 0
CPtoCP = []

for a in nodes:
    CPto = []
    for b in nodes:
        if a != b:
            dis = e_path[(a, b)][1]
        else:
            dis = 1
        CPto.append(dis)
        print(count_a, count_b, dis)
        count_b += 1
    CPtoCP.append(CPto)
    count_a += 1
    count_b = 0

with open('CP_info.csv', 'w') as g:
    writer = csv.writer(g)
    writer.writerows(CPtoCP)

print('Fin.')
