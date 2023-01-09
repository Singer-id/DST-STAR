import numpy as np
import json
import csv
from torch.utils.data import Dataset
import torch
import random
import re
import os
from copy import deepcopy


def slot_recovery(slot):
    if "pricerange" in slot:
        return slot.replace("pricerange", "price range")
    elif "arriveby" in slot:
        return slot.replace("arriveby", "arrive by")
    elif "leaveat" in slot:
        return slot.replace("leaveat", "leave at")
    else:
        return slot


class Processor(object):
    def __init__(self, config):
        # MultiWOZ dataset
        if "data/mwz" in config.data_dir:
            fp_ontology = open(os.path.join(config.data_dir, "ontology-modified.json"), "r")
            ontology = json.load(fp_ontology)
            fp_ontology.close()
        else:
            raise NotImplementedError()

        self.ontology = ontology
        self.slot_meta = list(self.ontology.keys()) # must be sorted
        self.num_slots = len(self.slot_meta)
        self.slot_idx = [*range(0, self.num_slots)] #*是解包的意思
        self.label_list = [self.ontology[slot] for slot in self.slot_meta]
        self.label_map = [{label: i for i, label in enumerate(labels)} for labels in self.label_list]
        #print(self.label_map)
        self.config = config
        self.domains = sorted(list(set([slot.split("-")[0] for slot in self.slot_meta])))
        self.num_domains = len(self.domains)
        self.domain_slot_pos = [] # the position of slots within the same domain
        self.description = json.load(open("utils/slot_description.json", 'r'))

        cnt = {}
        for slot in self.slot_meta:
            domain = slot.split("-")[0]
            if domain not in cnt:
                cnt[domain] = 0
            cnt[domain] += 1
        st = 0
        for di, domain in enumerate(self.domains):
            self.domain_slot_pos.append(list(range(st, st+cnt[domain])))
            st += cnt[domain]
        
    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if len(line) > 0 and line[0][0] == '#':     # ignore comments (starting with '#')
                    continue
                lines.append(line)
            return lines

    def get_train_instances(self, data_dir, tokenizer):
        return self._create_instances(self._read_tsv(os.path.join(data_dir, "train.tsv")), tokenizer)

    def get_dev_instances(self, data_dir, tokenizer):
        return self._create_instances(self._read_tsv(os.path.join(data_dir, "dev.tsv")), tokenizer)

    def get_test_instances(self, data_dir, tokenizer):
        return self._create_instances(self._read_tsv(os.path.join(data_dir, "test.tsv")), tokenizer)


    def _create_instances(self, lines, tokenizer):
        instances = []
        last_uttr = None
        last_dialogue_state = {}
        history_uttr = []

        #lines = lines[9074:9094]
        for (i, line) in enumerate(lines[0:10]):
            dialogue_idx = line[0]
            turn_idx = int(line[1])
            is_last_turn = (line[2] == "True")
            system_response = line[3]
            user_utterance = line[4]
            turn_dialogue_state = {}
            turn_dialogue_state_ids = []

            for idx in self.slot_idx:
                turn_dialogue_state[self.slot_meta[idx]] = line[5+idx]
                turn_dialogue_state_ids.append(self.label_map[idx][line[5+idx]])

            if turn_idx == 0: # a new dialogue
                last_dialogue_state = {}
                history_uttr = []
                history_type_turn_id = {} # json {adj:[],num:[]} 手动打标签
                history_token_turn_id = []
                last_uttr = ""
                for slot in self.slot_meta:
                    last_dialogue_state[slot] = "none"
                
            turn_only_label = [] # turn label
            for s, slot in enumerate(self.slot_meta):
                if last_dialogue_state[slot] != turn_dialogue_state[slot]:
                    turn_only_label.append(slot + "-" + turn_dialogue_state[slot])
                    #打标签
                    if turn_dialogue_state[slot] not in ["none","true","false","do not care"] :
                        value_type = self.description[slot]["value_type"]
                        if value_type not in history_type_turn_id: #添加key，初始化value
                            history_type_turn_id[value_type] = []

                        if turn_idx not in history_type_turn_id[value_type]: #构建实体类型_turn列表
                            history_type_turn_id[value_type].append(turn_idx)

            history_uttr.append(last_uttr)

            text_a = (system_response + " " + user_utterance).strip()
            text_b = ' '.join(history_uttr) # 所有历史的拼接

            last_uttr = text_a

            #手动make_instance(考虑train 和 test时make_instance的情况，设计好之后把函数提出去
            max_seq_length = self.config.max_seq_length

            if max_seq_length is None:
                max_seq_length = self.max_seq_length

            state = []
            for slot in self.slot_meta:
                s = slot_recovery(slot)
                k = s.split('-')
                v = last_dialogue_state[slot].lower()  # use the original slot name as index
                if v == "none":
                    continue
                k.extend([v])  # without symbol "-"
                t = tokenizer.tokenize(' '.join(k))
                state.extend(t)

            if not state:
                state.extend("null")

            avail_length_1 = max_seq_length - 3
            diag_1 = tokenizer.tokenize(text_a)
            diag_2 = tokenizer.tokenize(text_b)

            diag_1_length = len(diag_1)
            diag_2_length = len(diag_2)
            avail_length = avail_length_1 - diag_1_length

            token_turn_id = [turn_idx] * diag_1_length

            if diag_2_length > avail_length:  # truncated
                avail_length = diag_2_length - avail_length
                diag_2 = diag_2[avail_length:]
                history_temp = history_token_turn_id[avail_length:]
                input_token_turn_id = [-1] + history_temp + [-1] + token_turn_id + [-1]
            elif (diag_2_length == 0) and (diag_1_length > avail_length_1):
                avail_length = diag_1_length - avail_length_1
                diag_1 = diag_1[avail_length:]
                temp = token_turn_id[avail_length:]
                input_token_turn_id = [-1] + history_token_turn_id + [-1] + temp + [-1]
            else:
                input_token_turn_id = [-1] + history_token_turn_id + [-1] + token_turn_id + [-1]

            # we keep the order
            #drop_mask = [0] + [1] * diag_2_length + [0] + [1] * diag_1_length + [0]  # word dropout
            #TODO drop_mask
            diag_2 = ["[CLS]"] + diag_2 + ["[SEP]"]
            diag_1 = diag_1 + ["[SEP]"]
            diag = diag_2 + diag_1

            '''
            # word dropout
            if self.config.word_dropout > 0.:
                drop_mask = np.array(drop_mask)
                word_drop = np.random.binomial(drop_mask.astype('int64'), self.config.word_dropout)
                diag = [w if word_drop[i] == 0 else '[UNK]' for i, w in enumerate(diag)]
            '''
            input_id = tokenizer.convert_tokens_to_ids(diag)
            segment_id = [0] * len(diag_2) + [1] * len(diag_1)
            input_mask = [1] * len(diag)

            input_id_state = tokenizer.convert_tokens_to_ids(state)
            segment_id_state = [0] * len(state)
            input_mask_state = [1] * len(state)


            '''
            if len(self.input_id) != len(input_token_turn_list):
                file = open('train_bug.txt', mode='a+')
                file.write("dialogue_id:" + str(dialogue_idx) + "  turn_id:" + str(turn_idx) + '\n')
                file.close()
                continue
            '''
            # ID, turn_id, turn_utter, dialogue_history, label_ids,
            # turn_label, curr_turn_state, last_turn_state,
            # history_type_turn_id, input_token_turn_id,
            # input_id, segment_id, input_mask, input_id_state, segment_id_state, input_mask_state,
            # max_seq_length, slot_meta, is_last_turn, ontology

            instance = TrainingInstance(dialogue_idx, turn_idx, text_a, text_b, turn_dialogue_state_ids,
                                        turn_only_label, turn_dialogue_state, last_dialogue_state,
                                        history_type_turn_id, input_token_turn_id,
                                        input_id, segment_id, input_mask, input_id_state, segment_id_state, input_mask_state,
                                        self.config.max_seq_length, self.slot_meta, is_last_turn, self.ontology)
            instances.append(instance)
            last_dialogue_state = turn_dialogue_state
            history_token_turn_id += token_turn_id

            #print("——————————————")
            #print(diag_1)
            #print(token_turn_id)
            #print(diag_2)
            #print(history_token_turn_id)
            #print(dialogue_idx)
            #print(diag)
            #print(input_token_turn_id)
            #print("^^^^")
            #print(len(diag))
            #print(len(input_token_turn_id))


        return instances
            

class TrainingInstance(object):
    def __init__(self, ID, turn_id, turn_utter, dialogue_history, label_ids,
                 turn_label, curr_turn_state, last_turn_state,
                 history_type_turn_id, input_token_turn_id,
                 input_id, segment_id, input_mask, input_id_state, segment_id_state, input_mask_state,
                 max_seq_length, slot_meta, is_last_turn, ontology):

        self.dialogue_id = ID
        self.turn_id = turn_id
        self.turn_utter = turn_utter
        self.dialogue_history = dialogue_history
        
        self.curr_dialogue_state = curr_turn_state
        self.last_dialogue_state = last_turn_state
        self.gold_last_state = deepcopy(last_turn_state)
        
        self.turn_label = turn_label
        self.label_ids = label_ids

        self.history_type_turn_id = history_type_turn_id
        self.input_token_turn_id = input_token_turn_id

        self.input_id =  input_id
        self.segment_id = segment_id
        self.input_mask = input_mask

        self.input_id_state = input_id_state
        self.segment_id_state = segment_id_state
        self.input_mask_state = input_mask_state
       
        self.max_seq_length = max_seq_length
        self.slot_meta = slot_meta
        self.is_last_turn = is_last_turn
        
        self.ontology = ontology

    def renew_instance_state(self, tokenizer):
        state = []
        for slot in self.slot_meta:
            s = slot_recovery(slot)
            k = s.split('-')
            v = self.last_dialogue_state[slot].lower()  # use the original slot name as index
            if v == "none":
                continue
            k.extend([v])  # without symbol "-"
            t = tokenizer.tokenize(' '.join(k))
            state.extend(t)
        if not state:
            state.extend("null")

        self.input_id_state = tokenizer.convert_tokens_to_ids(state)
        self.segment_id_state = [0] * len(state)
        self.input_mask_state = [1] * len(state)


class MultiWozDataset(Dataset):
    def __init__(self, data, tokenizer, word_dropout=0., state_dropout=0.):
        self.data = data
        self.len = len(data)
        self.tokenizer = tokenizer
        self.word_dropout = word_dropout
        self.state_dropout = state_dropout
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        #self.data[idx].make_instance(self.tokenizer, word_dropout=self.word_dropout, state_dropout=self.state_dropout)
        return self.data[idx]

    def collate_fn(self, batch):
        def padding(list1, list2, list3, pad_token):
            max_len = max([len(i) for i in list1]) # utter-len
            result1 = torch.ones((len(list1), max_len)).long() * pad_token
            result2 = torch.ones((len(list2), max_len)).long() * pad_token
            result3 = torch.ones((len(list3), max_len)).long() * pad_token
            for i in range(len(list1)):
                result1[i, :len(list1[i])] = list1[i]
                result2[i, :len(list2[i])] = list2[i]
                result3[i, :len(list3[i])] = list3[i]
            return result1, result2, result3
        
        input_ids_list, segment_ids_list, input_mask_list = [], [], []
        input_ids_state_list, segment_ids_state_list, input_mask_state_list = [], [], []
        input_token_turn_list, history_type_turn_id_list = [], []

        for f in batch:
            input_ids_list.append(torch.LongTensor(f.input_id))
            segment_ids_list.append(torch.LongTensor(f.segment_id))
            input_mask_list.append(torch.LongTensor(f.input_mask))

            input_ids_state_list.append(torch.LongTensor(f.input_id_state))
            segment_ids_state_list.append(torch.LongTensor(f.segment_id_state))
            input_mask_state_list.append(torch.LongTensor(f.input_mask_state))

            input_token_turn_list.append(f.input_token_turn_id)
            history_type_turn_id_list.append(f.history_type_turn_id) #[{adj:[],type:[]}{adj:[]}] 没有to tensor

        input_ids, segment_ids, input_mask = padding(input_ids_list, segment_ids_list, input_mask_list, torch.LongTensor([0]))
        input_ids_state, segment_ids_state, input_mask_state = padding(input_ids_state_list, segment_ids_state_list,
                                                                       input_mask_state_list, torch.LongTensor([0]))
        label_ids = torch.tensor([f.label_ids for f in batch], dtype=torch.long)

        # padding
        max_len = max([len(i) for i in input_ids_list])
        result1 = np.ones((len(input_token_turn_list), max_len), dtype=int) * (-1)
        for i in range(len(input_token_turn_list)):
            result1[i, :len(input_token_turn_list[i])] = input_token_turn_list[i]
        input_token_turn_list = result1

        #history_type_turn_id_list = np.array(history_type_turn_id_list)

        return input_ids, segment_ids, input_mask, input_ids_state, segment_ids_state, input_mask_state, label_ids, \
               input_token_turn_list, history_type_turn_id_list