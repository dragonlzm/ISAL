"""This file is to collect all results in one floder, and save results in a csv file and plot all results.
Directory structure like: /cifar10-active-learning/202102230205-10 202102230406-20 202102230910-30 ...

A example: python collect_folder_results.py ----work-dir /cifar10-active-learning

"""

import os
import json
import math
import argparse
import colorlover as cl
import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='gather\save\plot')
    parser.add_argument('--work-dir', type=str, help='where save csv file and png file')
    args = parser.parse_args()
    return args


def get_plot_colors(max_colors, color_format="pyplot"):
    """auto Generate colors for plotting.
    param:
        max_colors: num of color class
    return:
        colors: different color list
    """
    colors = cl.scales["11"]["qual"]["Paired"]
    if max_colors > len(colors):
        colors = cl.to_rgb(cl.interp(colors, max_colors))
    if color_format == "pyplot":
        return [[j / 255.0 for j in c] for c in cl.to_numeric(colors)]
    return colors


def gather_save_csv(work_dir):
    """gather from every step floder and save active learning results to csv file
    param:
        work_dir: dir where save csv file and step floder
    """
    indices = []
    data_list = []
    columns_name_list = []
    i = 0
    # read data from every step floder
    for floder in os.listdir(work_dir):
        floder = os.path.join(work_dir, floder)
        if not os.path.isdir(floder):
            continue
        csv_file_path = os.path.join(floder, "summary.csv")
        if os.path.exists(csv_file_path):
            print("read summary from", floder)
        else:
            print("not found sum file in ", floder)
            continue
        dataFrame = pd.read_csv(csv_file_path)
        data = dataFrame.values
        # del first colums which is method name
        data = data[...,1:][0]
        index_name = dataFrame.iloc[0][0]
        print("mining_method", index_name)
        if i == 0:
            columns_name_list = list(map(float,dataFrame.columns.values.tolist()[1:]))
            columns_name_list = list(map(int,columns_name_list))
        # add values
        data_list.append(data)
        indices.append(index_name)
    # save csv file
    print(data_list)
    print(indices)
    print(columns_name_list)
    new_dataFrame = pd.DataFrame(data_list, columns=columns_name_list,index=indices)
    csv_file_path = os.path.join(work_dir, "summary.csv")
    new_dataFrame.to_csv(csv_file_path)

    return csv_file_path


def plot_acc_curves(csv_file_path, png_file_path):
    """Plot accuracy curves using matplotlib.pyplot and save to file.
    param:
        csv_file_path: csv file path which contains all expr results
        png_file_path: saved image file path
    """
    dataFrame = pd.read_csv(csv_file_path)
    row_num = len(dataFrame)
    colors = get_plot_colors(row_num)
    index_name_list = [dataFrame.iloc[i][0] for i in range(row_num)]
    columns_name_list = list(map(float,dataFrame.columns.values.tolist()[1:]))
    columns_name_list = [math.ceil(num) for num in columns_name_list]
    row_list = [dataFrame.iloc[i].tolist()[1:] for i in range(row_num)]

    #plot every row in results
    for i in range(row_num):
        plt.plot(columns_name_list,row_list[i] , 'o-', c=colors[i], alpha=0.8, label=str(index_name_list[i]))

    plt.title("accuarcy" + " vs. training labeled sample\n", fontsize=14)
    plt.xlabel("labeled sample", fontsize=14)
    plt.ylabel("acc", fontsize=14)
    plt.grid(alpha=0.4)
    plt.legend(fontsize=8)
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    plt.savefig(png_file_path)
    plt.clf()


def main():
    args = parse_args()
    print(args.work_dir)
    # gather all method results and save as one csv file
    csv_file_path = gather_save_csv(args.work_dir)
    png_path = os.path.join(args.work_dir, "results.png")
    plot_acc_curves(csv_file_path, png_path)

if __name__ == '__main__':
    main()