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
    return init_data


def init(data_set_dir):
    old_data_file = data_set_dir + "trace2650.txt"
    new_data_file = data_set_dir + "trace2650_preprocessed.csv"
    if os.path.exists(new_data_file):
        return pd.read_csv(new_data_file)
    data = preprocess_data(old_data_file, new_data_file)
    return data


my_data = init(data_set_dir_)
# my_data_groupeby_sd = my_data.groupby(["s", "d"])

my_data_nu = my_data.to_numpy()


#print("Size of flux {}".format(my_data.groupby('fib').count()))
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

    print("--- %s seconds ---" % (end_time - start_time))
    save_glob_stats(nodes)


def save_glob_stats(nodes):
    ret = []
    for node in nodes:
        sub_ret = [node, nodes[node]["dropped_by_me"], nodes[node]["queue_size"]]
        ret.append(sub_ret)
    r = pd.DataFrame(ret)
    r.to_csv("glob", sep=" ", index=False)


def cal_std(group_data, mean):
    return 1


def cal_rtt_time(data):
    print("Start !")
    start_time = time.time()
    packets = {}
    means = {}
    # time,code,pid,fid,s,d,pos
    for line in data:
        code = line[1]
        pid = line[2]
        time_ = line[0]
        src = line[4]
        dst = line[5]
        packet_name = "{}".format(pid)

        is_ = packet_name in packets
        if code == 0:
            if not is_:
                packets[packet_name] = {}
            packets[packet_name]["start"] = time_
            continue
        elif code == 3:
            if not is_:
                packets[packet_name] = {}
            packets[packet_name]["end"] = time_
        else:
            continue
        entry_f1 = "{}{}".format(src, dst)
        entry_f2 = "{}{}".format(dst, src)

        if "end" in packets[packet_name] and "start" in packets[packet_name]:
            if entry_f1 in means:
                Xi = float(packets[packet_name]["end"] - packets[packet_name]["start"])
                means[entry_f1]["val"] += Xi
                means[entry_f1]["vari"] += Xi ** 2
                means[entry_f1]["count"] += 1
                # print("Xi = {}\tXi^2 = {}".format(Xi,Xi**2))

            elif entry_f2 in means:
                Xi = float(packets[packet_name]["end"] - packets[packet_name]["start"])
                means[entry_f2]["val"] += Xi
                means[entry_f2]["vari"] += Xi ** 2
                means[entry_f2]["count"] += 1
            else:
                Xi = float(packets[packet_name]["end"] - packets[packet_name]["start"])
                means[entry_f1] = {}
                means[entry_f1]["val"] = Xi
                means[entry_f1]["vari"] = Xi ** 2
                means[entry_f1]["count"] = 1
    end_time = time.time()
    print("--- %s seconds ---" % (end_time - start_time))
    print("Count : {}".format(len(means)))
    countAll = 0
    meanAll = 0
    Xi_All = 0 #square !
    for end_to_end in means:
        countAll += 1
        count = float(means[end_to_end]["count"])
        val = float(means[end_to_end]["val"])
        X_bar = float(val / count)
        meanAll += X_bar
        Xi_All += float(X_bar)**2
        means[end_to_end]["mean"] = X_bar
        means[end_to_end]["vari"] = means[end_to_end]["vari"] - count * (X_bar ** 2)
        means[end_to_end]["vari"] = float(means[end_to_end]["vari"]) / float(count - 1)
        var = means[end_to_end]["vari"]
        std = var ** (float(1) / float(2))
        seg_DN = std / float(count)
        print(" {} -----> Mean : {} \tConfidence Interval 68%  [ {}   ,    {}   ]".
              format(end_to_end, X_bar, std - seg_DN, std + seg_DN))

    meanAll = float(meanAll) / float(countAll)
    varAll = (Xi_All - countAll*(meanAll**2)) / float(countAll-1)
    segAll = varAll**(float(1) / float(2))
    segAll_DN = segAll / float(countAll)
    print(" [+]-----------> MeanAll : {} \tConfidence Interval 68%  [ {}   ,    {}   ]".
      format(meanAll, segAll - segAll_DN, segAll + segAll_DN))



# test(my_data_nu)
cal_rtt_time(my_data_nu)

# calc_global_data(my_data_nu)
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
# pg.c('plot "tmp.dat" using 1:2 w linespoint')
# pg.c('plot "tmp.dat" using 1:2 w linespoint title "xyz"')
# pg.c("set xlabel 'lala' ")
# pg.c("set ylabel 'lili' ")
#
#
# %%

# W = np.arange(20, 30)
# Z = Y ** 6.0
# pg.s([X, Y])

# pg.c('plot "tmp.dat" using 1:2 w linespoint title "xyz"')
# pg.c("save 'plot test saving'")
