"""This file is to gather results in all step folder, and save it in a csv file and plot results.

A example: python $(dirname "$0")/src/tools/plot_results.py \
    --work-dir=${WORK_DIR}/${TIMESTAMP} --train-step=${TRAIN_STEP}
    
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
    parser.add_argument('--train-step', type=int, default=10, help='active learning training step')
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


def gather_save_csv(work_dir, step_num):
    """gather from every step floder and save active learning results to csv file
    param:
        work_dir: dir where save csv file and step floder
        step_num: active learning training step
    """
    indices = []
    data_list = []
    columns_name_list = []
    # read data from every step floder
    for i in range(step_num):
        step_dir = os.path.join(work_dir, "step_" + str(i))
        json_file_path = os.path.join(step_dir, "results.json")
        with open(json_file_path, 'r') as load_f:
            load_dict = json.load(load_f)
            columns_name_list.append(load_dict["cur_data_num"])
            data_list.append(load_dict["al_result"])
            if i == 0:
                indices = [load_dict["mining_method"]]
    data_list = [data_list]
    # save csv file
    dataFrame = pd.DataFrame(data_list, columns=columns_name_list,index=indices)
    csv_file_path = os.path.join(work_dir, "summary.csv")
    dataFrame.to_csv(csv_file_path)

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
        plt.plot(columns_name_list,row_list[i] , '-', c=colors[i], alpha=0.8, label=str(index_name_list[i]))

    plt.title("accuarcy" + " vs. training labeled sample\n", fontsize=14)
    plt.xlabel("labeled sample", fontsize=14)
    plt.ylabel("acc", fontsize=14)
    plt.grid(alpha=0.4)
    plt.legend()
    plt.savefig(png_file_path)
    plt.clf()


def main():
    args = parse_args()
    csv_file_path = gather_save_csv(args.work_dir, args.train_step)
    png_path = os.path.join(args.work_dir, "results.png")
    plot_acc_curves(csv_file_path, png_path)

if __name__ == '__main__':
    main()
