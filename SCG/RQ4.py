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
repeat_time = 1
t_start = time.time()
device = 'cuda'
for project in project_list:
    for random_seed in random_seed_list:
        filename = f"RQ4/{project}_{random_seed}-noSMOTE.txt"
            
        if os.path.exists(filename):
            print(f"{filename} already exists, skipping...")
            continue
        t_round = time.time()
        command = f"python train-noSMOTE.py --project {project} --random_seed {random_seed} --repeat_time {repeat_time} --device {device}"
            
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

    f1_score1 = extract_metrics(results, "F1-Score1")

    return f1_score1

f1_scores = {}

for project in project_list:
    f1_score1_list = []
    for random_seed in random_seed_list:
        filename = f"RQ4/{project}_{random_seed}-noSMOTE.txt"
        f1_score1 = process_file(filename)
        
        if f1_score1 is not None:
            f1_score1_list.append(f1_score1)

    f1_scores[project] = statistics.mean(f1_score1_list)

    print(f"Test set results for {project}:")

    print("Average F1_score1: {:.2f}".format(f1_scores[project]))
    print()



