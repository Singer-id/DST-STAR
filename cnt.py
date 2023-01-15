import os
import json
import re
import argparse

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]
data_files = ["train_dials_v2.json", "dev_dials_v2.json", "test_dials_v2.json"]
data_dir = "data/mwz2.1"

def get_label_type_dict(slot_meta, slot_type, ontology):
    # 辅助BIO的
    label_type_dict = {}  # 按type打的候选值集合
    for idx, s in enumerate(slot_meta):
        t = slot_type[idx]
        if t not in label_type_dict:
            label_type_dict[t] = []
        label_type_dict[t].extend(ontology[s])

    for k in label_type_dict:
        label_type_dict[k] = list(set(label_type_dict[k]))  # 去重

    return label_type_dict

def read_json(file_name):
    fp_data = open(os.path.join(data_dir, file_name), "r")
    dials = json.load(fp_data)
    return dials


if "data/mwz" in data_dir:
    fp_ontology = open(os.path.join(data_dir, "ontology-modified.json"), "r")
    ontology = json.load(fp_ontology)
    fp_ontology.close()
else:
    raise NotImplementedError()

ontology  = ontology
slot_meta = list(ontology.keys())  # must be sorted
description = json.load(open("utils/slot_description.json", 'r'))
slot_type = [description[slot]["value_type"] for slot in slot_meta]
label_type_dict = get_label_type_dict(slot_meta, slot_type, ontology)

for i in range(len(data_files)):
    dials = read_json(data_files[i])
    #cnt = 0
    cnt_all = 0 #所有数据有多少条
    cnt_by_type = {}
    for dial in dials:
        for turn in dial["dialogue"]:
            #cnt_all += 1
            turn_idx = turn["turn_idx"]
            if turn_idx == 0:
                input_text = ""
            turn_input = turn["system_transcript"] + " " + turn["transcript"]
            input_text += turn_input
            for list in turn["turn_label"]:
                slot = list[0]
                value = list[1]
                if slot not in description:
                    continue
                cnt_all += 1
                type = description[slot]["value_type"]
                if value not in input_text: #有一个匹配不上
                    if type not in cnt_by_type:
                        cnt_by_type[type] = 0
                    cnt_by_type[type] += 1
                    #cnt += 1

    #print(data_files[i].split(".")[0] + "_bugs :" + str(cnt))
    print(data_files[i].split(".")[0] + "_all :" + str(cnt_all))
    print(data_files[i].split(".")[0] + "_by_type :" + json.dumps(cnt_by_type))
