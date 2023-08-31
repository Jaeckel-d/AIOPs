import os
import tqdm
import time
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from utils import Logger, check_dir
from multiprocessing import Pool, Queue

logger = Logger("./run/preprocess.log")

foi = ["dt", "serial_number", "smart_1_normalized", "smart_4raw", "smart_5raw",
       "smart_7_normalized", "smart_9_normalized", "smart_10_normalized", "smart_12raw",
       "smart_187_normalized", "smart_194_normalized",
       "smart_197raw", "smart_198raw", "smart_199raw"]
fnf = ["smart_1_normalized", "smart_4raw", "smart_5raw",
       "smart_7_normalized", "smart_9_normalized", "smart_10_normalized", "smart_12raw",
       "smart_187_normalized", "smart_194_normalized",
       "smart_197raw", "smart_198raw", "smart_199raw"]
LOG_THRESH = 30
mt_queue = Queue()


def process_one_month(log_root_path, csv_name, target_path):
    msg = "Initalize {} processing, dispatched to pid: {}".format(csv_name, os.getpid())
    logger.log(msg)

    one_time_start = time.time()

    csv_path = os.path.join(log_root_path, csv_name)
    df = pd.read_csv(csv_path)
    length = len(df)

    msg = "processing: {}, {} logs in it".format(csv_name, length)
    logger.log(msg)

    df = df[foi]
    df[fnf] = df[fnf].fillna(df[fnf].mean())

    target_dir = os.path.join(target_path, csv_name.split(".")[0])
    check_dir(target_dir)

    serials = df["serial_number"].value_counts()
    scnt = len(serials)
    serials = serials[serials >= LOG_THRESH]
    rmvd = scnt - len(serials)
    msg = "Remove {} disks with logs under {} entries (in {})".format(rmvd, LOG_THRESH, csv_name)
    logger.log(msg)

    for serial, count in zip(serials.index, serials):
        serial_df = df[df["serial_number"] == serial].drop(columns="serial_number")
        csv_name = "{}.csv".format(serial)
        csv_path = os.path.join(target_dir, csv_name)
        serial_df.sort_values(by="dt", inplace=True)
        serial_df.to_csv(csv_path, index=False)

    msg = "{} completed, time used: {}s".format(csv_name, time.time() - one_time_start)
    logger.log(msg)

    mt_queue.put(length)


def process_all_mt(log_root_path, target_path, pool_size=4):
    check_dir(target_path)

    msg = "Start to process disk logs in multi-process"
    logger.log(msg)
    mt_pool = Pool(pool_size)

    all_time_start = time.time()
    for csv_name in os.listdir(log_root_path):
        mt_pool.apply_async(process_one_month, args=(log_root_path, csv_name, target_path))

    mt_pool.close()
    mt_pool.join()

    total_logs = 0
    while not mt_queue.empty():
        total_logs += mt_queue.get(True)

    msg = "All logs completed, total logs: {}, time used: {}s".format(total_logs, time.time() - all_time_start)
    logger.log(msg)


def merge_months_logs(src_path, target_path):
    check_dir(target_path)
    seen_flgs = np.zeros(250000, dtype=np.uint8)
    logger.log("Starting to merge all month logs")
    total_logs = 0
    start_time = time.time()
    for dir_name in os.listdir(src_path):
        logger.log("Processing: {}".format(dir_name))
        dir_path = os.path.join(src_path, dir_name)
        for disk_csv_name in os.listdir(dir_path):
            disk_csv_path = os.path.join(dir_path, disk_csv_name)
            df = pd.read_csv(disk_csv_path)
            total_logs += len(df)
            disk_name = disk_csv_name.split(".")[0]
            disk_id = int(disk_name.split("_")[1])
            target_csv_path = os.path.join(target_path, disk_csv_name)
            if seen_flgs[disk_id] == 0:
                seen_flgs[disk_id] = 1
            else:
                df_prev = pd.read_csv(target_csv_path)
                df = pd.concat([df_prev, df])
                df.sort_values(by="dt", inplace=True)
            df.to_csv(target_csv_path, index=False)
    logger.log("All month processed, total logs read: {},time uesd: {}s"
               .format(total_logs, time.time() - start_time))


def extract_log_info(fault_tag_path, log_path):
    fault_tag_df = pd.read_csv(fault_tag_path)
    fault_tag_df["fault_time"] = pd.to_datetime(fault_tag_df["fault_time"], format="%Y-%m-%d").dt.strftime("%Y%m%d")
    print(fault_tag_df.head())

    log_list = [x.split(".")[0] for x in os.listdir(log_path)]
    fault_list = [1 if fault_tag_df["serial_number"].isin([x]).any() else 0 for x in log_list]

    # if treat as multi-class classification, use below statement
    # just change fault label from [0, 1] to [0, ...n] (n denote categories of faults)
    # fault_list = [int(fault_tag_df[fault_tag_df["serial_number"] == x]["tag"].values[0]) + 1
    #               if fault_tag_df["serial_number"].isin([x]).any() else 0 for x in log_list]

    date_list = [fault_tag_df[fault_tag_df["serial_number"] == x]["fault_time"].values[0]
                 if fault_tag_df["serial_number"].isin([x]).any() else '0' for x in log_list]

    df = pd.DataFrame({
        "serial_number": log_list,
        "fault": fault_list,
        "fault_date": date_list,
    })
    df.to_csv("data/disk_info.csv", index=False)


if __name__ == "__main__":

    # process_all_mt(log_root_path="data/smartlog_data/", target_path="data/processed_mt/")
    # merge_months_logs(src_path="data/processed_mt", target_path="data/preprocessed")
    extract_log_info(fault_tag_path="data/fault_tag_data.csv", log_path="data/preprocessed")
