import os
import tqdm
import time
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from utils import Logger

logger = Logger("./run/preprocess.log")

def plotDiskLogCount(df_serials):
    plt.figure(figsize=(10, 6))
    bins = [0, 50, 100, 150, 200, 250, 300, 350, 400]
    serials_seg = pd.cut(df_serials, bins, right=False)
    counts = serials_seg.value_counts(sort=False)

    plt.title("number of disks for log ranges ")
    plt.xlabel("number of logs(range)")
    plt.ylabel("disk counts")
    plt.bar(counts.index.astype(str), counts)

    plt.show()


def plotColumnDistribution(df):
    df = df.drop(columns=["serial_number"])
    # pick columns that have between 1 and 50 unique values
    columnNames = list(df)
    nGraphCol, nGraphRow = 4, 3
    nGraph = nGraphCol * nGraphRow
    print(nGraph, nGraphCol, nGraphRow)
    plt.figure(num=None, figsize=(3 * nGraphCol, 4 * nGraphRow), dpi=100, facecolor='w', edgecolor='k')
    for i in range(1, nGraph+1):
        plt.subplot(nGraphRow, nGraphCol, i)
        columnDf = df.iloc[:, i]
        if not np.issubdtype(type(columnDf.iloc[0]), np.number):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation=90)
        plt.title(f'{columnNames[i]} (column {i})')
        plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    # plt.show()
    plt.savefig("DistributionofFoi.png", dpi=120)


# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum=1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title("Correlation Matrix", fontsize=15)
    plt.savefig("Correlation.png", dpi=120)


def plotfailure(df):
    failure = df["failure"].value_counts()

    plt.title("Good/Failure Disk")
    plt.pie(failure.values, labels=["Good", "failure"], colors=["gold", "lightcoral"],
            autopct="%.2f%%", explode=[0, 0.1], shadow=True)

    # plt.show()
    plt.savefig("diskProportion.png", dpi=120)



def processSerials(df):
    serials = df["serial_number"].value_counts()
    total = len(serials)
    # vis_disk_log_counts(serials)
    cnt = 0
    for serial, count in zip(serials.index, serials):
        serial_df = df[df["serial_number"] == serial].drop(columns="serial_number")
        csv_name = "{}.csv".format(serial)
        serial_df.to_csv(os.path.join(target_dir, csv_name), index=False)
        cnt += 1
        print("\r processed file: {}/{}".format(cnt, total), end="")
        # do not uitlize disks log number under 50
        if count < 50:
            break


def processFailure(df: pd.DataFrame):
    serials = df["serial_number"].value_counts()
    serials = serials[serials >= 50]

    temp = df[["serial_number", "failure"]]
    # print(len(serials))
    temp = temp.groupby("serial_number").max()
    temp = temp[temp.index.isin(serials.index)]
    # print(temp.info())
    # print(temp.head())
    temp.to_csv("./data/hasFailure.csv")


def processDiskData(df, target_dir="data/ST4000/", log_thresh=10):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
    disk_log_path = os.path.join(target_dir, "preprocessed/")
    if not os.path.exists(disk_log_path):
        os.makedirs(disk_log_path, exist_ok=True)

    # scnt = len(df)
    # # drop disks only has failure logs
    # drop_list = df.groupby("serial_number").min()
    # drop_list = drop_list[drop_list["failure"] == 1]
    # df = df[~df["serial_number"].isin(drop_list.index)]
    # rmvd = scnt - len(df)
    # print("Remove {} logs which belongs to disks only have failure recorded.".format(rmvd))

    serials = df["serial_number"].value_counts()
    scnt = len(serials)
    serials = serials[serials >= log_thresh]
    rmvd = scnt - len(serials)
    print("Remove {} disks with logs under {} entries".format(rmvd, log_thresh))

    # blacklist = checkInvalid("data/ST4000/preprocessed/")

    total = len(serials)
    cnt = 0
    for serial, count in zip(serials.index, serials):
        # if blacklist.isin(serial).any():
        #     continue
        serial_df = df[df["serial_number"] == serial].drop(columns="serial_number")
        csv_name = "{}.csv".format(serial)
        serial_df.to_csv(os.path.join(disk_log_path, csv_name), index=False)
        cnt += 1
        print("\r processed file: {}/{}".format(cnt, total), end="")
        # do not uitlize disks log number under 50

    # temp = df[["serial_number", "failure"]]
    # # print(len(serials))
    #
    # temp = temp.groupby("serial_number").max()
    # temp = temp[temp.index.isin(serials.index)]
    # # temp = temp[~temp["serial_number"].isin(blacklist["serial_number"])]
    #
    # # print(temp.info())
    # # print(temp.head())
    # temp.to_csv(os.path.join(target_dir, "hasFailure.csv"))
    # print(temp["failure"].value_counts())

def checkInvalid(path):
    blacklist = []
    for file in os.listdir(path):
        df = pd.read_csv("{}/{}".format(path, file))
        df.sort_values(["date", "failure"], inplace=True)
        flg1 = False
        for idx, row in df.iterrows():
            if row["failure"] == 1:
                flg1 = True
            if row["failure"] == 0 and flg1:
                blacklist.append(file.split(".")[0])
                print(file)
                break
    blacklist = {"serial_number": blacklist}
    return pd.DataFrame(blacklist)


def processFaultTag(path):
    df = pd.read_csv(path)
    print(df.head())
    df["fault_time"] = pd.to_datetime(df["fault_time"], format="%Y-%m-%d").dt.strftime("%Y%m%d")
    print(df.sort_values(by="fault_time"))


def process_one_csv(df, seen_flag, target_dir="data/processed/disks/", log_thresh=30, first_time=False, ):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    serials = df["serial_number"].value_counts()
    scnt = len(serials)
    serials = serials[serials >= log_thresh]
    rmvd = scnt - len(serials)
    msg = "Remove {} disks with logs under {} entries".format(rmvd, log_thresh)
    logger.log(msg)

    total = len(serials)
    cnt = 0
    for serial, count in zip(serials.index, serials):
        # if blacklist.isin(serial).any():
        #     continue
        serial_df = df[df["serial_number"] == serial].drop(columns="serial_number")
        csv_name = "{}.csv".format(serial)
        csv_path = os.path.join(target_dir, csv_name)
        disk_id = int(serial.split("_")[1])
        if seen_flag[disk_id] == 1:
            before_df = pd.read_csv(csv_path)
            serial_df = pd.concat([before_df, serial_df])
        serial_df.sort_values(by="dt", inplace=True)
        serial_df.to_csv(csv_path, index=False)
        seen_flag[disk_id] = 1
        cnt += 1
        print("\r processed file: {}/{}".format(cnt, total), end="")
        print("")
    return seen_flag


def process_all(log_root_path, target_path="data/processed/disks/"):
    msg = "Start to Process disk logs"
    logger.log(msg)

    foi = ["dt", "serial_number", "smart_1_normalized", "smart_4raw", "smart_5raw",
           "smart_7_normalized", "smart_9_normalized", "smart_10_normalized", "smart_12raw",
           "smart_187_normalized", "smart_194_normalized",
           "smart_197raw", "smart_198raw", "smart_199raw"]
    fnf = ["smart_1_normalized", "smart_4raw", "smart_5raw",
           "smart_7_normalized", "smart_9_normalized", "smart_10_normalized", "smart_12raw",
           "smart_187_normalized", "smart_194_normalized",
           "smart_197raw", "smart_198raw", "smart_199raw"]
    total_logs = 0
    flg = [0 for i in range(500000)]
    all_time_start = time.time()
    for csv_name in os.listdir(log_root_path):
        one_time_start = time.time()
        csv_path = os.path.join(log_root_path, csv_name)
        df = pd.read_csv(csv_path)
        length = len(df)

        msg = "Processing: {}, {} logs in it".format(csv_name, length)
        logger.log(msg)

        df = df[foi]
        df[fnf] = df[fnf].fillna(df[fnf].mean())
        flg = process_one_csv(df, flg, target_path, first_time=(total_logs == 0), )
        total_logs += length

        msg = "{} completed, time used: {}s".format(csv_name, time.time() - one_time_start)
        logger.log(msg)

    msg = "All logs completed, total logs: {}, time used: {}s".format(total_logs, time.time() - all_time_start)
    logger.log(msg)


if __name__ == "__main__":

    process_all(log_root_path="data/smartlog_data/", target_path="data/processed/")

    # foi = ["dt", "serial_number", "smart_1_normalized", "smart_4raw", "smart_5raw",
    #        "smart_7_normalized", "smart_9_normalized", "smart_10_normalized", "smart_12raw",
    #        "smart_187_normalized", "smart_194_normalized",
    #        "smart_197raw", "smart_198raw", "smart_199raw"]
    #
    # # target_dir = "data/ST4000/preprocessed/"
    #
    # file_path = "data/smartlog_data/smartlog_data_201801.csv"
    #
    # target_dir = "data/{}/".format("201801")
    #
    # year = 2018
    # df = pd.read_csv(file_path)
    # # for year in range(2016, 2019):
    # #     tmp_df = pd.read_csv(file_path.format(year))
    # #     df = pd.concat([df, tmp_df])
    # df = df[foi]
    # print("Total log entries: ", df.__len__())
    #
    # fnf = ["smart_1_normalized", "smart_4raw", "smart_5raw",
    #        "smart_7_normalized", "smart_9_normalized", "smart_10_normalized", "smart_12raw",
    #        "smart_187_normalized", "smart_194_normalized",
    #        "smart_197raw", "smart_198raw", "smart_199raw"]
    # df[fnf] = df[fnf].fillna(df[fnf].mean())
    #
    # processDiskData(df, target_dir)
    # checkInvalid("data/2015/preprocessed")

    # processFaultTag("data/fault_tag_data.csv")
