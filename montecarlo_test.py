import random

cities = 32
num = 300
trials = 10000

def generate():
    list = []
    for i in range(num):
        list.append(random.randint(0,cities-1))
    return list

counter = 0
for i in range(trials):
    test = generate()
    check_list = [0]*cities
    for a in test:
        check_list[a] = 1
    #print(sum(check_list))
    if sum(check_list) == cities:
        counter += 1

rate = counter / trials
print(rate)