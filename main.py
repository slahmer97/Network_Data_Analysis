import pandas as pd
import numpy as np
import os.path
import time
import PyGnuplot as pg

data_set_dir_ = "/home/slahmer/PycharmProjects/Network_Data_Analysis/data_set/"


def preprocess_data(data_origin, dest_file):
    init_data = pd.read_csv(data_origin, sep=" ")
    print("[+] Origin file contains {} entries".format(len(init_data)))
    init_data.loc[init_data.code == 4, 'pos'] = init_data.loc[init_data.code == 4, 'd']
    init_data.loc[init_data.code == 4, 'd'] = init_data.loc[init_data.code == 4, 's']
    init_data.loc[init_data.code == 4, 's'] = init_data.loc[init_data.code == 4, 'bif']
    init_data = init_data.drop(columns=['tos', 'bif'])
    init_data.to_csv(dest_file, index=False)
    print("[+] New file contains {} entries".format(len(init_data)))
    return init_data.to_numpy()


def init(data_set_dir):
    old_data_file = data_set_dir + "trace2650.txt"
    new_data_file = data_set_dir + "trace2650_preprocessed.csv"
    if os.path.exists(new_data_file):
        return pd.read_csv(new_data_file)
    data = preprocess_data(old_data_file, new_data_file)
    return data


my_data = init(data_set_dir_)
my_data_nu = my_data.to_numpy()


# groupe by source,dest see time end-to-ennd
# select where code == 4 groupeby pos average teaux de perte
# ...

# %%
# test = my_data.groupby(['s', 'd'])


# 0 : départ de la source
# 1 : arrivée dans un nœud intermédiaire 
# 2 : départ d’une file d’attente
# 3 : arrivée à destination
# 4 : destruction d’un paquet (placement dans une file pleine)

def get_end_to_end_time(df):
    start = -1
    end = -1
    for index, line in df.iterrows():
        if line["code"] == 3:
            end = line["time"]
        if line["code"] == 0:
            start = line["time"]

    if start < 0 or end < 0:
        return -1
    return end - start




def calc_global_data(data):
    print("Start !")
    print("Len ", len(data))

    nodes = {}
    for i in range(1, 27):
        tmp = "N{}".format(i)
        nodes[tmp] = {}
        nodes[tmp]["passed_by_me"] = 0
        nodes[tmp]["processed_by_me"] = 0
        nodes[tmp]["dropped_by_me"] = 0
        nodes[tmp]["transmitted_by_me"] = 0
        nodes[tmp]["queue_size"] = 0
        nodes[tmp]["curr_queue_size"] = 0

    total_send_packet = 0
    total_lost_packet = 0
    total_arrived_packet = 0
    start_time = time.time()
    counter = 0
    # time,code,pid,fid,s,d,pos
    for line in data:
        node = line[6]
        if line[1] == 0:  # départ de la source
            total_send_packet += 1
            nodes[node]["transmitted_by_me"] += 1
            nodes[node]["processed_by_me"] += 1
        elif line[1] == 1:  # arrivée dans un nœud intermédiaire
            nodes[node]["curr_queue_size"] += 1
            a = nodes[node]["curr_queue_size"]
            b = nodes[node]["queue_size"]
            c = max(a, b)
            nodes[node]["queue_size"] = c
            # print("Curr : ", a, "-------old size : ", b, "------new size : ", c)
        elif line[1] == 2:  # départ d’une file d’attente
            nodes[node]["processed_by_me"] += 1  # TODO RE-check
            # nodes[node]["curr_queue_size"] -= 1
        elif line[1] == 3:  # arrivée à destination
            total_arrived_packet += 1
        elif line[1] == 4:  # destruction d’un paquet (placement dans une file pleine)
            total_lost_packet += 1
            nodes[node]["dropped_by_me"] += 1
            nodes[node]["curr_queue_size"] -= 1
    end_time = time.time()
    print("End !")
    print("[+] Total number of sent packet : {}".format(total_send_packet))
    print("[+] Total number of lost packet : {}".format(total_lost_packet))

    print("[+] Total number of arrived packet : {}".format(total_arrived_packet))
    print("[+] Infos")
    for n in nodes:
        print(nodes[n])
    print("--- %s seconds ---" % (end_time - start_time))


calc_global_data(my_data_nu)
# means = []
# end_to_end = []
# for name, groupe in test:
#    mean_tmp = 0
#    counter = 0
#    data_groupe_by_pid = groupe.groupby("pid")
#    for name2, groupe2 in data_groupe_by_pid:
#        tmp = get_end_to_end_time(groupe2)
#        if tmp >= 0:
#            counter = counter + 1
#            mean_tmp = mean_tmp + tmp
# mu = mean_tmp / counter
# means.append(mean_tmp / counter)
# end_to_end.append(name)
# print(name," ===> ",mu)


# len_ = len(means)
# for i in range(0, len_):
#    print(end_to_end[i], "   ", means[i])

# %%
X = np.arange(10)
Y = np.sin(X/(2*np.pi))
Z = Y**2.0
pg.s([X,Y,Z])
pg.c('plot "tmp.dat" u 1:2 w lp')
pg.c('replot "tmp.dat" u 1:3 w lp')
pg.p('myfigure.ps')