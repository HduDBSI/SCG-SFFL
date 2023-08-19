# Investigating the impact of training epochs and feature vector dimensions on the word2vec model.

import os
import time
import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np
import statistics

project_list = ['activemq', 'alluxio', 'binnavi', 'kafka', 'realm-java']
random_seed_list = list(range(5))
repeat_time =5
t_start = time.time()
device = 'cuda'
for project in project_list:
    for random_seed in random_seed_list:
        filename = f"RQ4/Ours/{project}_{random_seed}.txt"
            
        if os.path.exists(filename):
            print(f"{filename} already exists, skipping...")
            continue
        t_round = time.time()
        command = f"python train.py --project {project} --random_seed {random_seed} --repeat_time {repeat_time} --device {device}"
            
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        stdout, stderr = process.communicate()

        with open(filename, "w") as file:
            file.write(stdout.decode())
            print(f'Results have been saved to {filename}')
        print('This round time:', time.time() - t_round)

print('Total time:', time.time() - t_start)


def process_file(filename):
    def extract_metrics(content, metric_name):
        match = re.search(f"{metric_name}:\s+(\d+\.\d+)%", content)
        if match:
            return float(match.group(1))
        return None
    
    if not os.path.exists(filename):
        return None

    with open(filename, "r") as file:
        content = file.read()

    start_index = content.find("Test set\n") + len("Test set\n")
    results = content[start_index:]

    precision1 = extract_metrics(results, "Precision1")
    recall1 = extract_metrics(results, "Recall1")
    f1_score1 = extract_metrics(results, "F1-Score1")

    precision2 = extract_metrics(results, "Precision2")
    recall2 = extract_metrics(results, "Recall2")
    f1_score2 = extract_metrics(results, "F1-Score2")

    return precision1, recall1, f1_score1, precision2, recall2, f1_score2



def calculate_metrics(folder, project_list, random_seed_list):
    precisions = {}
    recalls = {}
    f1_scores = {}

    for project in project_list:
        precision1_list = []
        recall1_list = []
        f1_score1_list = []
        precision2_list = []
        recall2_list = []
        f1_score2_list = []
        
        for random_seed in random_seed_list:
            filename = f"RQ4/{folder}/{project}_{random_seed}.txt"
            precision1, recall1, f1_score1, precision2, recall2, f1_score2 = process_file(filename)

            if precision1 is not None:
                precision1_list.append(precision1)

            if recall1 is not None:
                recall1_list.append(recall1)

            if f1_score1 is not None:
                f1_score1_list.append(f1_score1)

            if precision2 is not None:
                precision2_list.append(precision2)

            if recall2 is not None:
                recall2_list.append(recall2)

            if f1_score2 is not None:
                f1_score2_list.append(f1_score2)

        precisions[(project, '1')] = statistics.mean(precision1_list)
        recalls[(project, '1')] = statistics.mean(recall1_list)
        f1_scores[(project, '1')] = statistics.mean(f1_score1_list)

        precisions[(project, '2')] = statistics.mean(precision2_list)
        recalls[(project, '2')] = statistics.mean(recall2_list)
        f1_scores[(project, '2')] = statistics.mean(f1_score2_list)

        print(f"Test set results for {project}:")
        print("Average Precision1: {:.2f}".format(precisions[(project, '1')]))
        print("Average Recall1: {:.2f}".format(recalls[(project, '1')]))
        print("Average F1_score1: {:.2f}".format(f1_scores[(project, '1')]))

        print("Average Precision2: {:.2f}".format(precisions[(project, '2')]))
        print("Average Recall2: {:.2f}".format(recalls[(project, '2')]))
        print("Average F1_score2: {:.2f}".format(f1_scores[(project, '2')]))
        print()

    return precisions, recalls, f1_scores

ours_precisions, ours_recalls, ours_f1_scores = calculate_metrics('SCG', project_list, random_seed_list)
yu_precisions, yu_recalls, yu_f1_scores = calculate_metrics('SFFL', project_list, random_seed_list)

from tabulate import tabulate

# 定义表头
headers = [
    'Project', 'Approach',
    r'$\textit{precision}_1$', r'$\textit{recall}_1$', r'$\textit{F}_1\text{-score}_1$', r'$\textit{accuracy}$'  
    r'$\textit{precision}_2$', r'$\textit{recall}_2$', r'$\textit{F}_1\text{-score}_2$',
]

# 创建一个空的结果列表
table_data = []

# 遍历每个项目
project_dic = {'binnavi': 'BinNavi', 'activemq': 'ActiveMQ', 'kafka': 'Kafka', 'alluxio': 'Alluxio', 'realm-java': 'Realm-java'}
for project in project_list:
    
    ours_precision1, ours_recall1, ours_f1_score1 = ours_precisions[(project, '1')], ours_recalls[(project, '1')], ours_f1_scores[(project, '1')]
    ours_precision2, ours_recall2, ours_f1_score2 = ours_precisions[(project, '2')], ours_recalls[(project, '2')], ours_f1_scores[(project, '2')]

    yu_precision1, yu_recall1, yu_f1_score1 = yu_precisions[(project, '1')], yu_recalls[(project, '1')], yu_f1_scores[(project, '1')]
    yu_precision2, yu_recall2, yu_f1_score2 = yu_precisions[(project, '2')], yu_recalls[(project, '2')], yu_f1_scores[(project, '2')]

    ours_acc = ours_precision2 / ours_precision1 * 100
    yu_acc = yu_precision2 / yu_precision1 * 100

    max_precision1 = max(ours_precision1, yu_precision1)
    max_recall1 = max(ours_recall1, yu_recall1)
    max_f1_score1 = max(ours_f1_score1, yu_f1_score1)

    max_precision2 = max(ours_precision2, yu_precision2)
    max_recall2 = max(ours_recall2, yu_recall2)
    max_f1_score2 = max(ours_f1_score2, yu_f1_score2)

    max_acc = max(ours_acc, yu_acc)

    table_data.append([
        r'\multirow{2}{*}{' + project_dic[project] + r'}',
        r'SFFL',
        r'\textbf{' + '{:.2f}'.format(ours_precision1) + r'\%}' if ours_precision1 == max_precision1 else '{:.2f}'.format(ours_precision1) + r'\%',
        r'\textbf{' + '{:.2f}'.format(ours_recall1) + r'\%}' if ours_recall1 == max_recall1 else '{:.2f}'.format(ours_recall1) + r'\%',
        r'\textbf{' + '{:.2f}'.format(ours_f1_score1) + r'\%}' if ours_f1_score1 == max_f1_score1 else '{:.2f}'.format(ours_f1_score1) + r'\%',
        r'\textbf{' + '{:.2f}'.format(ours_acc) + r'\%}' if ours_acc == max_acc else '{:.2f}'.format(ours_acc) + r'\%',
        r'\textbf{' + '{:.2f}'.format(ours_precision2) + r'\%}' if ours_precision2 == max_precision2 else '{:.2f}'.format(ours_precision2) + r'\%',
        r'\textbf{' + '{:.2f}'.format(ours_recall2) + r'\%}' if ours_recall2 == max_recall2 else '{:.2f}'.format(ours_recall2) + r'\%',
        r'\textbf{' + '{:.2f}'.format(ours_f1_score2) + r'\%}' if ours_f1_score2 == max_f1_score2 else '{:.2f}'.format(ours_f1_score2) + r'\%',
    ])

    table_data.append([
        r' ',
        r'SCG',
        r'\textbf{' + '{:.2f}'.format(yu_precision1) + r'\%}' if yu_precision1 == max_precision1 else '{:.2f}'.format(yu_precision1) + r'\%',
        r'\textbf{' + '{:.2f}'.format(yu_recall1) + r'\%}' if yu_recall1 == max_recall1 else '{:.2f}'.format(yu_recall1) + r'\%',
        r'\textbf{' + '{:.2f}'.format(yu_f1_score1) + r'\%}' if yu_f1_score1 == max_f1_score1 else '{:.2f}'.format(yu_f1_score1) + r'\%',
        r'\textbf{' + '{:.2f}'.format(yu_acc) + r'\%}' if yu_acc == max_acc else '{:.2f}'.format(yu_acc) + r'\%',
        r'\textbf{' + '{:.2f}'.format(yu_precision2) + r'\%}' if yu_precision2 == max_precision2 else '{:.2f}'.format(yu_precision2) + r'\%',
        r'\textbf{' + '{:.2f}'.format(yu_recall2) + r'\%}' if yu_recall2 == max_recall2 else '{:.2f}'.format(yu_recall2) + r'\%',
        r'\textbf{' + '{:.2f}'.format(yu_f1_score2) + r'\%}' if yu_f1_score2 == max_f1_score2 else '{:.2f}'.format(yu_f1_score2) + r'\%',
    ])

ours_average_precision1 = statistics.mean([ours_precisions[(project, '1')] for project in project_list])
ours_average_recall1 = statistics.mean([ours_recalls[(project, '1')] for project in project_list])
ours_average_f1_score1 = statistics.mean([ours_f1_scores[(project, '1')] for project in project_list])
ours_average_acc = statistics.mean([ours_precisions[(project, '2')] / ours_precisions[(project, '1')] * 100 for project in project_list])
ours_average_precision2 = statistics.mean([ours_precisions[(project, '2')] for project in project_list])
ours_average_recall2 = statistics.mean([ours_recalls[(project, '2')] for project in project_list])
ours_average_f1_score2 = statistics.mean([ours_f1_scores[(project, '2')] for project in project_list])

yu_average_precision1 = statistics.mean([yu_precisions[(project, '1')] for project in project_list])
yu_average_recall1 = statistics.mean([yu_recalls[(project, '1')] for project in project_list])
yu_average_f1_score1 = statistics.mean([yu_f1_scores[(project, '1')] for project in project_list])
yu_average_acc = statistics.mean([yu_precisions[(project, '2')] / yu_precisions[(project, '1')] * 100 for project in project_list])
yu_average_precision2 = statistics.mean([yu_precisions[(project, '2')] for project in project_list])
yu_average_recall2 = statistics.mean([yu_recalls[(project, '2')] for project in project_list])
yu_average_f1_score2 = statistics.mean([yu_f1_scores[(project, '2')] for project in project_list])

max_avg_precision1 = max(ours_average_precision1, yu_average_precision1)
max_avg_recall1 = max(ours_average_recall1, yu_average_recall1)
max_avg_f1_score1 = max(ours_average_f1_score1, yu_average_f1_score1)
max_avg_acc = max(ours_average_acc, yu_average_acc)
max_avg_precision2 = max(ours_average_precision2, yu_average_precision2)
max_avg_recall2 = max(ours_average_recall2, yu_average_recall2)
max_avg_f1_score2 = max(ours_average_f1_score2, yu_average_f1_score2)

# Add average row to table data
table_data.append([
    r'\multirow{2}{*}{\textbf{Average}}',
    r'SFFL',
    r'\textbf{' + '{:.2f}'.format(ours_average_precision1) + r'\%}' if ours_average_precision1 == max_avg_precision1 else '{:.2f}'.format(ours_average_precision1) + r'\%',
    r'\textbf{' + '{:.2f}'.format(ours_average_recall1) + r'\%}' if ours_average_recall1 == max_avg_recall1 else '{:.2f}'.format(ours_average_recall1) + r'\%',
    r'\textbf{' + '{:.2f}'.format(ours_average_f1_score1) + r'\%}' if ours_average_f1_score1 == max_avg_f1_score1 else '{:.2f}'.format(ours_average_f1_score1) + r'\%',
    r'\textbf{' + '{:.2f}'.format(ours_average_acc) + r'\%}' if ours_average_acc == max_avg_acc else '{:.2f}'.format(ours_average_acc) + r'\%',
    r'\textbf{' + '{:.2f}'.format(ours_average_precision2) + r'\%}' if ours_average_precision2 == max_avg_precision2 else '{:.2f}'.format(ours_average_precision2) + r'\%',
    r'\textbf{' + '{:.2f}'.format(ours_average_recall2) + r'\%}' if ours_average_recall2 == max_avg_recall2 else '{:.2f}'.format(ours_average_recall2) + r'\%',
    r'\textbf{' + '{:.2f}'.format(ours_average_f1_score2) + r'\%}' if ours_average_f1_score2 == max_avg_f1_score2 else '{:.2f}'.format(ours_average_f1_score2) + r'\%',
])

table_data.append([
    r' ',
    r'SCG',
    r'\textbf{' + '{:.2f}'.format(yu_average_precision1) + r'\%}' if yu_average_precision1 == max_avg_precision1 else '{:.2f}'.format(yu_average_precision1) + r'\%',
    r'\textbf{' + '{:.2f}'.format(yu_average_recall1) + r'\%}' if yu_average_recall1 == max_avg_recall1 else '{:.2f}'.format(yu_average_recall1) + r'\%',
    r'\textbf{' + '{:.2f}'.format(yu_average_f1_score1) + r'\%}' if yu_average_f1_score1 == max_avg_f1_score1 else '{:.2f}'.format(yu_average_f1_score1) + r'\%',
    r'\textbf{' + '{:.2f}'.format(yu_average_acc) + r'\%}' if yu_average_acc == max_avg_acc else '{:.2f}'.format(yu_average_acc) + r'\%',
    r'\textbf{' + '{:.2f}'.format(yu_average_precision2) + r'\%}' if yu_average_precision2 == max_avg_precision2 else '{:.2f}'.format(yu_average_precision2) + r'\%',
    r'\textbf{' + '{:.2f}'.format(yu_average_recall2) + r'\%}' if yu_average_recall2 == max_avg_recall2 else '{:.2f}'.format(yu_average_recall2) + r'\%',
    r'\textbf{' + '{:.2f}'.format(yu_average_f1_score2) + r'\%}' if yu_average_f1_score2 == max_avg_f1_score2 else '{:.2f}'.format(yu_average_f1_score2) + r'\%',
])

# 将表格数据转换为LaTeX表格
latex_table = tabulate(table_data, headers, tablefmt='latex')
latex_table = latex_table.replace('\\textbackslash{}', '').replace('\\}', '}').replace('\\{', '{').replace('text', '\\text').replace('\\$', '$').replace('\\_','_').replace('multirow', '\\multirow')
# 打印LaTeX表格
print(latex_table)