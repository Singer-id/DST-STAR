import os
import json
import re
import argparse


data_files = ["train_dials_v2.json", "dev_dials_v2.json", "test_dials_v2.json"]

class CrfDataProcessor(object):
    def __init__(self, args):
        if "data/mwz" in args.data_dir:
            fp_ontology = open(os.path.join(args.data_dir, "ontology-modified.json"), "r")
            ontology = json.load(fp_ontology)
            fp_ontology.close()
        else:
            raise NotImplementedError()

        self.ontology  = ontology
        self.slot_meta = list(ontology.keys())  # must be sorted
        self.description = json.load(open("utils/slot_description.json", 'r'))
        self.slot_type = [self.description[slot]["value_type"] for slot in self.slot_meta]
        self.label_type_dict = self._get_label_type_dict()
        self.args = args

    def _get_label_type_dict(self):
        # 辅助BIO的
        label_type_dict = {}  # 按type打的候选值集合
        for idx, s in enumerate(self.slot_meta):
            t = self.slot_type[idx]
            if t not in label_type_dict:
                label_type_dict[t] = []
            label_type_dict[t].extend(self.ontology[s])

        for k in label_type_dict:
            label_type_dict[k] = list(set(label_type_dict[k]))  # 去重

        return label_type_dict

    def read_json(self, file_name):
        fp_data = open(os.path.join(args.data_dir, file_name), "r")
        dials = json.load(fp_data)
        return dials

    def re_tagging(self, input_text, type):
        # TODO 此策略在标test时使用，候选值集合中的a|b等特殊值直接加进去
        # type == num or time
        if type == 'num':
            pattern = "\s[0-9]\s"
        else:
            pattern = "([0-1]?[0-9]|2[0-3]):([0-5][0-9])"
        mappings = re.finditer(pattern, input_text)

        value_list = []
        value_pos_list = []
        for i in mappings:
            value_pos_list.append(i.span())
            if type == 'num':
                value_list.append(i.match()[1])
            else:
                value_list.append(i.match())

        # TODO 确定下标属于的轮次
        return value_list, value_pos_list

    def BIO_tagging(self, input_text, type):
        for value in
        mappings = re.finditer(pattern, input_text)


    def create_data(self, file_name, description, tokenizer):
        dials = self.read_json(file_name)
        dials_for_crf = []
        for dial in dials:
            dialogue_idx = dial["dialogue_idx"]
            domains = dial["domains"] #TODO

            for turn in dial["dialogue"]:
                turn_idx = turn["turn_idx"]
                if turn_idx == 0:
                    input_text = ""
                    input_turn_list = []
                    belief_state_dict = {}
                    belief_state_type_dict = {
                        "time":[],
                        "num":[],
                        "adj":[],
                        "day":[],
                        "area":[],
                        "location":[],
                        "food":[],
                        "type":[]
                    }
                turn_input = turn["system_transcript"] + " " + turn["transcript"]
                input_text += turn_input
                input_turn_list += [turn_idx] * len(turn_input)

                for list in turn["turn_label"]:
                    slot = list[0]
                    value = list[1]
                    type = description[slot]["value_type"]
                    #按槽名储存label
                    if value == 'none': #delete的情况
                        if slot in belief_state_dict:
                            del belief_state_dict[slot]
                    else:
                        belief_state_dict[slot] = value
                    #按类型储存label
                    belief_state_type_dict[type].append(value) #出现过的label全部加进去

                # BIO tagging 优先级
                # 1.time num #TODO 生成或稳定加入类似3|1的特殊值
                # 2.adj day area location food, type(用另一个模型训)
                # 先标优先级低的，再标优先级高的

                #tokenize
                diag = tokenizer.tokenize(input_text) # token list

                #标注location
                for t, value_list in enumerate(self.label_type_dict["location"]):
                    for value in value_list:
                        value_token = tokenizer.tokenize(value)
                        temp = [1 * (token in diag) for token in value_token]





                next((i for i, x in enumerate(myList) if x), None)

                self.BIO_tagging(input_text)








                new_dial = {
                    "dialogue_idx":dialogue_idx,
                    "domains":domains,
                    "turn_idx":turn_idx,
                    "belief_state_dict":belief_state_dict,
                    "belief_state_type_dict":belief_state_type_dict
                }
                dials_for_crf.append(new_dial)
        #写入
        with open(os.path.join(data_dir, file_name.split(".")[0] + "_v3.json"), 'w') as outfile:
            json.dump(dials_for_crf, outfile, indent=4)

def main(args):
    if "data/mwz" in args.data_dir:
        fp_ontology = open(os.path.join(args.data_dir, "ontology-modified_v2.json"), "r")
        ontology = json.load(fp_ontology)
        fp_ontology.close()
    else:
        raise NotImplementedError()

    slot_meta = list(ontology.keys())  # must be sorted
    description = json.load(open("utils/slot_description.json", 'r'))

    for i in len(data_files):
        create_data(args.data_dir,data_files[i],description)

if "__name__" == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='data/mwz2.1')  # data/mwz2.4
    args = parser.parse_args()
    main(args)
