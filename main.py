import pandas as pd
import numpy as np
import os.path
import time
import PyGnuplot as pg

import matplotlib.pyplot as plt

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


# fl = len(my_data.fid.unique())
# print("Flux count : {}".format(fl))
# exit(1)


# my_data_nu = my_data.to_numpy()

# d_time = len(my_data.time.unique())
# print("Unique time : ", d_time)


# print("Size of flux {}".format(my_data.groupby('fib').count()))
# groupe by source,dest see time end-to-ennd
# select where code == 4 groupeby pos average teaux de perte


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
        sub_ret = [node, nodes[node]["dropped_by_me"], nodes[node]["queue_size"], nodes[node]["curr_queue_size"]]
        ret.append(sub_ret)
    r = pd.DataFrame(ret, columns=["node", "drop", "queue_s", "curr_queue_s"])
    r.to_csv("glob.csv", sep=",", index=False)


def plot_pert_prop(file_name):
    # width of the bars
    barWidth = 0.2
    tmp = pd.read_csv(file_name)
    x = tmp.NODE.to_numpy()
    proportion_de_pert = tmp.A.to_numpy() * 100
    taux_de_perte = tmp.B.to_numpy() * 100
    plt.bar(x, proportion_de_pert, label='proportion de perte', color='r')
    # plt.bar(x,taux_de_perte,  label='taux de perte',color='c')
    plt.xlabel('Node')
    plt.ylabel('Loss percentage')
    # plt.title('Exemple d\' histogramme simple')
    plt.show()


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
    Xi_All = 0  # square !
    bar = []
    err = []
    name = []
    for end_to_end in means:
        name.append(end_to_end)
        countAll += 1
        count = float(means[end_to_end]["count"])
        val = float(means[end_to_end]["val"])
        X_bar = float(val / count)
        meanAll += X_bar
        Xi_All += float(X_bar) ** 2
        means[end_to_end]["mean"] = X_bar
        means[end_to_end]["vari"] = means[end_to_end]["vari"] - count * (X_bar ** 2)
        means[end_to_end]["vari"] = float(means[end_to_end]["vari"]) / float(count - 1)
        var = means[end_to_end]["vari"]
        std = var ** (float(1) / float(2))
        seg_DN = std / float(count)
        bar.append(X_bar)
        err.append(seg_DN * 3)
        # print(" {} -----> Mean : {} \tConfidence Interval 68%  [ {}   ,    {}   ]".
        #      format(end_to_end, X_bar, std - 3*seg_DN, std + 3*seg_DN))

    meanAll = float(meanAll) / float(countAll)
    varAll = (Xi_All - countAll * (meanAll ** 2)) / float(countAll - 1)
    segAll = varAll ** (float(1) / float(2))
    segAll_DN = segAll / float(countAll)
    name.append("All")
    bar.append(meanAll)
    err.append(segAll_DN * 3)
    print(" [+]-----------> MeanAll : {} \tConfidence Interval 68%  [ {}   ,    {}   ]".
          format(meanAll, segAll - 3 * segAll_DN, segAll + 3 * segAll_DN))
    # plot_conf_int(bar, err, name)


# cal_rtt_time(my_data_nu)

def get_empty_node_info(node):
    nodes_info = {}
    n_pseudo = node
    nodes_info[n_pseudo] = {}
    nodes_info[n_pseudo]["current_queue_size"] = 0
    nodes_info[n_pseudo]["loss_packet"] = 0
    nodes_info[n_pseudo]["passed_by_me"] = 0
    return nodes_info


def cal_net_stats(data, file_name, disp=False):
    res = []
    print("start")
    start_time = time.time()
    counter = 0
    # time,code,pid,fid,s,d,pos
    last_index: int = -1
    for line in data:
        time_ = line[0]
        code_ = line[1]
        if 0 <= last_index < len(res) and 'time' in res[last_index] and res[last_index]['time'] == time_:
            if code_ == 4:  # destruction d’un paquet (placement dans une file pleine)
                res[last_index]["lost_packet"] += 1
                res[last_index]["packets_in_network"] -= 1
            elif code_ == 0:  # depart de source
                res[last_index]["injected_packet"] += 1
                res[last_index]["packets_in_network"] += 1
            elif code_ == 3:  # arrivé au destination
                res[last_index]["packets_in_network"] -= 1

        else:
            index = last_index + 1
            res.append([])
            res[index] = {}
            res[index]["time"] = time_
            if index >= 1:
                res[index]["lost_packet"] = res[index - 1]["lost_packet"]
                res[index]["injected_packet"] = res[index - 1]["injected_packet"]
                res[index]["packets_in_network"] = res[index - 1]["packets_in_network"]

            else:
                res[index]["lost_packet"] = 0
                res[index]["injected_packet"] = 0
                res[index]["packets_in_network"] = 0

            if code_ == 4:  # destruction d’un paquet (placement dans une file pleine)
                res[index]["lost_packet"] += 1
                res[index]["packets_in_network"] -= 1
            elif code_ == 0:  # depart du source
                res[index]["injected_packet"] += 1
                res[index]["packets_in_network"] += 1
            elif code_ == 3:  # arrivé au destination
                res[index]["packets_in_network"] -= 1

            last_index += 1
    end_time = time.time()
    print("--- %s seconds ---" % (end_time - start_time))
    file = open("{}.csv".format(file_name), mode="w")
    file.write("time,lost_p,inject_p,live_p\n")
    for i in range(0, len(res)):
        file.write("{},{},{},{}\n".format(res[i]["time"], res[i]["lost_packet"], res[i]["injected_packet"],
                                          res[i]["packets_in_network"]))
        if disp:
            print("[+] at time %2.8f -- dropped : %6d -- injected : %10d -- live : %6d" % (
                res[i]["time"], res[i]["lost_packet"], res[i]["injected_packet"], res[i]["packets_in_network"]))


def cal_node_stats(data, node, disp=False):
    # packets drop in each router
    # packets drop in all network
    # packets injected in network
    # packets in network
    res = []
    print("start")
    start_time = time.time()
    counter = 0
    # time,code,pid,fid,s,d,pos
    last_index = -1
    for line in data:
        time_ = line[0]
        code_ = line[1]
        pos_ = line[6]
        if 0 <= last_index < len(res) and 'time' in res[last_index] and res[last_index]['time'] == time_:
            if code_ == 4 and pos_ == node:  # destruction d’un paquet (placement dans une file pleine)
                res[last_index]["dropped_packet"] += 1
        else:
            index = last_index + 1
            res.append([])
            res[index] = {}
            res[index]["time"] = time_
            if index >= 1:
                res[index]["dropped_packet"] = res[index - 1]["dropped_packet"]
                res[index]["passed_by_me"] = res[index - 1]["passed_by_me"]
                res[index]["sent_by_me"] = res[index - 1]["sent_by_me"]
                res[index]["current_queue_size"] = res[index - 1]["current_queue_size"]
            else:
                res[index]["dropped_packet"] = 0
                res[index]["passed_by_me"] = 0
                res[index]["sent_by_me"] = 0
                res[index]["current_queue_size"] = 0

            if pos_ == node:
                if code_ == 4:  # destruction d’un paquet (placement dans une file pleine)
                    res[index]["dropped_packet"] += 1
                    res[index]["current_queue_size"] -= 1
                elif code_ == 0:  # départ de la source
                    res[index]["sent_by_me"] += 1
                    res[index]["passed_by_me"] += 1
                elif code_ == 1:  # arrivée dans un nœud intermédiai
                    res[index]["current_queue_size"] += 1
                elif code_ == 2:  # départ d’une file d’attent
                    res[index]["passed_by_me"] += 1
                    # res[index]["current_queue_size"] -= 1

            last_index += 1
    end_time = time.time()
    print("--- %s seconds ---" % (end_time - start_time))

    file = open("{}.csv".format(node), mode="w")
    file.write("time,sent,passed,queue,dropped\n")
    for index in range(0, len(res)):
        file.write("{},{},{},{},{}\n".format(res[index]["time"], res[index]["sent_by_me"], res[index]["passed_by_me"],
                                             res[index]["current_queue_size"], res[index]["dropped_packet"]))
        if disp == True:
            print("[+] time : %2.6f -- send : %8d  -- passed : %8d  -- queue : %8d  -- dropped : %8d" % (
                res[index]["time"], res[index]["sent_by_me"], res[index]["passed_by_me"],
                res[index]["current_queue_size"],
                res[index]["dropped_packet"]))
    file.close()


def flux__(data_n):
    grp = data_n.groupby(["fid"])
    f_list = {}
    for name, group in grp:
        f_list[name] = {}
        f_list[name]["start"] = group["time"].min()
        f_list[name]["end"] = group["time"].max()
        f_list[name]["p_count"] = len(group["pid"].unique())
        # print("Name : %s  -- start at : %2.8f  -- end at : %2.8f -- p_count : %8d "%(name, f_list[name]["start"],
        # f_list[name]["end"], f_list[name]["p_count"]))

    print("\n\n\n")
    act_f_l = []
    file = open("act_flux.csv", mode="w")
    file.write("time,f_count\n")
    for i in np.arange(0, 19, 0.001):
        act_f = {}
        act_f["time"] = i
        act_f["f_count"] = 0
        for f in f_list:
            if float(f_list[f]["start"]) <= float(i) <= float(f_list[f]["end"]):
                act_f["f_count"] += 1
        print("time : %2.4f  -- F_count : %4d " % (act_f["time"], act_f["f_count"]))
        file.write("{},{}\n".format(act_f["time"], act_f["f_count"]))
        act_f_l.append(act_f)
    file.close()


def end_to_end_delay_stream(data_pd, stream_id):
    print("===============Stream ({})=============".format(stream_id))
    data = data_pd.loc[data_pd.fid == stream_id].to_numpy()
    paquet_his = {}
    for line in data:
        t = line[0]
        code = line[1]
        pid = line[2]
        if pid in paquet_his:
            if code == 3:
                paquet_his[pid]["end"] = t
            elif code == 4:
                del paquet_his[pid]
        else:
            paquet_his[pid] = {}
            paquet_his[pid]["start"] = t
            paquet_his[pid]["end"] = 0

    file = open("flux_id__{}.csv".format(stream_id), mode="w")
    count = 0
    for pid in paquet_his:
        count += 1
        file.write("{},{}\n".format(count,paquet_his[pid]["end"]-paquet_his[pid]["start"]))
        print("pid : %s --- start : %2.8f --- end : %2.8f" % (pid, paquet_his[pid]["start"], paquet_his[pid]["end"]))

    # print(data)
    return 1


end_to_end_delay_stream(my_data, 1)
end_to_end_delay_stream(my_data, 2)
end_to_end_delay_stream(my_data, 3)
end_to_end_delay_stream(my_data, 4)
end_to_end_delay_stream(my_data, 5)
end_to_end_delay_stream(my_data, 6)
end_to_end_delay_stream(my_data, 10)
end_to_end_delay_stream(my_data, 100)
end_to_end_delay_stream(my_data, 150)
end_to_end_delay_stream(my_data, 300)
end_to_end_delay_stream(my_data, 450)
end_to_end_delay_stream(my_data, 800)
end_to_end_delay_stream(my_data, 900)
end_to_end_delay_stream(my_data, 1000)
end_to_end_delay_stream(my_data, 1200)


# plot_pert_prop(file_name="/home/slahmer/CLionProjects/sidahmedhmar/cmake-build-debug/prop_et_taux_perte.txt")

# flux__(my_data)

# calc_global_data(my_data_nu)
# cal_node_stats(my_data_nu, "N23")
# cal_net_stats(my_data_nu, "new_stats", False)
# W = np.arange(20, 30)
# Z = Y ** 6.0
# pg.s([X, Y])

# pg.c('plot "tmp.dat" using 1:2 w linespoint title "xyz"')
# pg.c("save 'plot test saving'")
