"""
This file provides functions for scripts for defining student ids
and video ids that will be used for train and test split.
The script should be run only once, since the video id is picked randomly.
"""
import csv
import random

# fixed data
file = "../data/osats_score.csv"
# Where these numbers are from will be written in the manuscript
amount_train_split_pre = {'novice': 78, 'intermediate': 25, 'proficient': 4}
amount_train_split_post = {'novice': 1, 'intermediate': 26, 'proficient': 80}

def _read_pre_post(type: str, file: str) -> dict:
    """Return a list of data read form given csv file that has
    type tag ('PRE' or 'POST'). The returned list contains
    list of vid_id and average grs.
    EX: {'AHO729': [13, 14, 15]}
    """
    output = {}
    with open(file, "r") as f:
        data = csv.reader(f)
        next(data)
        for item in data:
            if item[2] == type:
                if item[0] not in output:
                    output[item[0]] = [int(item[-1])]
                else:
                    output[item[0]].append(int(item[-1]))
    return output

def _calculate_avg_grs(vid_to_score: dict) -> dict:
    """Return a dict, which key is a vid_id and value is avg grs"""
    output = {}
    for item in vid_to_score:
        avg_grs = sum(vid_to_score[item])/3
        output[item] = avg_grs
    return output

def create_vid_id_split(type: str, file: str) -> dict:
    """Return a dict that looks like the following:
    {novice: [vid_ids], intermediate: [vid_ids], proficient: [vid_ids]}
    """
    vid_to_score = _read_pre_post(type, file)
    avg_grs_per_vid = _calculate_avg_grs(vid_to_score)
    output = {'novice': [], 'intermediate': [], 'proficient': []}
    for id in avg_grs_per_vid:
        if avg_grs_per_vid[id] >= 24:
            output['proficient'].append(id)
        elif avg_grs_per_vid[id] >= 16:
            output['intermediate'].append(id)
        else:
            output['novice'].append(id)
    return output

def split_train(all_data: dict, amount_split: dict) -> dict:
    """Return a tuple of the split of train and test"""
    train_data = {'novice': [], 'intermediate': [], 'proficient': []}
    for item in all_data:
        train_data[item] = random.sample(all_data[item], k=amount_split[item])
    return train_data

def split_test_based_on_train(all_data: dict, train_data: dict) -> dict:
    """Return a tuple of the split of train and test"""
    test_data = {'novice': [], 'intermediate': [], 'proficient': []}
    for item in all_data:
        for id in all_data[item]:
            if id not in train_data[item]:
                test_data[item].append(id)
    return test_data