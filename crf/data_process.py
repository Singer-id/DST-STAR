# Copyright (c) Facebook, Inc. and its affiliates

import json
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import ast
from tqdm import tqdm
import os
import random
from functools import partial
from utils.fix_label import fix_general_label_error
from utils.bio_matching import value_matching
from collections import OrderedDict
from Model_crf import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]
import datetime

random.seed(577)
HISTORY_MAX_LEN = 450
GPT_MAX_LEN = 1024

slot_list=["attraction-area",
                    "attraction-name",
                    "attraction-type",
                    "hotel-area",
                    "hotel-book day",
                    "hotel-book people",
                    "hotel-book stay",
                    "hotel-internet",
                    "hotel-name",
                    "hotel-parking",
                    "hotel-pricerange",
                    "hotel-stars",
                    "hotel-type",
                    "restaurant-area",
                    "restaurant-book day",
                    "restaurant-book people",
                    "restaurant-book time",
                    "restaurant-food",
                    "restaurant-name",
                    "restaurant-pricerange",
                    "taxi-arriveby",
                    "taxi-departure",
                    "taxi-destination",
                    "taxi-leaveat",
                    "train-arriveby",
                    "train-book people",
                    "train-day",
                    "train-departure",
                    "train-destination",
                    "train-leaveat"]

def finetune_data(args, path_name):
    target_data=[]
    with open(path_name) as f:
        dials = json.load(f)
        for dial_dict in dials:
            if args["only_domain"] != "none" and args["only_domain"] in dial_dict["domains"]:
                target_data.append(dial_dict)

            random.Random(args["seed"]).shuffle(dials)
            target_data = target_data[:int(len(dials)*args["fewshot"])]
    return target_data

class DSTDataset(Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, args):
        """Reads source and target sequences from txt files."""
        self.data = data
        self.args = args

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item_info = self.data[index]
       
        return item_info

    def __len__(self):
        return len(self.data)

def turn_slot_values_find(pre_turn_slot_values,turn_belief_list):
    result={}
    for key in turn_belief_list.keys():
        # generate
        if key not in pre_turn_slot_values.keys():
            result[key]=turn_belief_list[key]
        else:
            # update
            if pre_turn_slot_values[key]!=turn_belief_list[key]:
                result[key]=turn_belief_list[key]

    return result

def state_process_24(input_list):
    result={}
    for slot in input_list:
        result[slot['slots'][0][0]]=slot['slots'][0][1]
    return result

def state_filter(state):
    result=[]
    for slot in state:
        # print( slot['slots'][0][0])
        if slot['slots'][0][0].split('-')[0] in EXPERIMENT_DOMAINS:
            result.append(slot)
    return result
def turn_label_process(input_list):
    result=[]
    for slot in input_list:
        if slot[0].split('-')[0] in EXPERIMENT_DOMAINS:
           result.append([slot[0],slot[1]])
    return result
def att_pre_turn_state_process(input_pre_turn_state,slot_temp):
    state={}
   
    for key in slot_temp:
        if key in input_pre_turn_state:
            state[key]=input_pre_turn_state[key]
        else:
            state[key]='none'
       
    return state
def read_data(args, path_name,dataset=None):
    slot_lang_list = ["description_human", "rule_description", "value_description", "rule2", "rule3"]
    co_reference_dict={}
    print(("Reading all files from {}".format(path_name)))
    data = []
    domain_counter = {}
    value_type_counter={}
    # read files
    data=[]
    with open(path_name) as f:
        dials = json.load(f)
        # print('dials',dials[0])

        if dataset=="train" and args["fewshot"]>0:
            dials = finetune_data(args, path_name)
        
        for dial_dict in tqdm(dials):
            dialog_history = ""
            if 'hospital' in dial_dict["domains"] or 'police' in dial_dict["domains"]:
                continue
            # Counting domains
            for domain in dial_dict["domains"]:
                if domain not in EXPERIMENT_DOMAINS:
                    continue
                if domain not in domain_counter.keys():
                    domain_counter[domain] = 0
                domain_counter[domain] += 1

            if args['use_2.4']:
                turns=enumerate(dial_dict["dialogue"])
            else:
                turns= enumerate(dial_dict["turns"])
            dialog_history={}
            for ti, turn in turns:
                # candidate_check(turn)
                # assert len(turn['input'].split())==len(turn['BIO_TAGGING'].split())
                # turn['input']=turn['input'].split()
                data.append(turn)
    return data
# def candidate_check(turn):
#     for key in turn['candiate_dict'].keys():
#         if len(turn['candiate_dict'][key])==0:
#             print('key',turn['turn_id'],key,turn['candiate_dict'][key])
def read_json_data(path_name,args):
    corasedata={}
    finedata={}
    result={}
    corasedata_list=[]
    finedata_list=[]
    with open(path_name) as f:
        dials = json.load(f)
        corasedata['pos_1']=dials['departure_dict']['train']+dials['departure_dict']['taxi']
        corasedata['neg_1'] = dials['destination_dict']['train'] + dials['destination_dict']['taxi']
        corasedata['pos_2'] = dials['arriveby_dict']['train'] + dials['arriveby_dict']['taxi']
        corasedata['neg_2'] = dials['leaveat_dict']['train'] + dials['leaveat_dict']['taxi']


        finedata['pos_1']=dials['departure_dict']['train']
        finedata['neg_1']= dials['destination_dict']['train']
        finedata['pos_2'] = dials['departure_dict']['taxi']
        finedata['neg_2'] = dials['destination_dict']['taxi']

        finedata['pos_3'] = dials['arriveby_dict']['train']
        finedata['neg_3'] = dials['leaveat_dict']['train']
        finedata['pos_4'] = dials['arriveby_dict']['taxi']
        finedata['neg_4'] = dials['leaveat_dict']['taxi']

    result['corase']=corasedata
    result['fine']=finedata
    length = max([len(corasedata['pos_1']), len(corasedata['neg_1']), len(corasedata['pos_2']), len(corasedata['neg_2'])])
    result=pre_train_data_fill(result,length,args)
    return result
    # return data

def pre_train_data_fill(pre_train_data,length,args):

    for type in pre_train_data.keys():
        for data_label in pre_train_data[type].keys():
            data_length=len(pre_train_data[type][data_label])
            if data_length==length:
                continue
            else:
                list=[]
                operation_num=int(length/data_length)
                for i in range(operation_num):
                    list.extend(pre_train_data[type][data_label])
                list.extend(pre_train_data[type][data_label][:length-data_length*operation_num])
                assert len(list)==length
                pre_train_data[type][data_label]=list
    data=batch_pre_train(pre_train_data,length,args)
    return data
def batch_pre_train(pre_train_data,length,args):
    data=[]
    name_list=[]
    contrastive_size=args['contrastive_size']
    for type in pre_train_data.keys():
        for data_label in pre_train_data[type].keys():
            name_list.append(f"{type}"+'-'+f"{data_label}")
    operation_num=int(length/contrastive_size)

    for i in range(operation_num):
        turn={}
        for name in name_list:
            turn[name]=pre_train_data[name.split('-')[0]][name.split('-')[1]][i*contrastive_size:(i+1)*contrastive_size]
        data.append(turn)
    # print(length,contrastive_size,operation_num,length-(operation_num * contrastive_size))
    turn = {}
    for name in name_list:

        turn[name] = pre_train_data[name.split('-')[0]][name.split('-')[1]][
                     operation_num * contrastive_size:length]
        data.append(turn)
    return data
def confusion_slot_collect(input_dict,slot1,slot2,turn_label,turn_dialogue):
    slot_exist_list=[]

    for data_idx,data in enumerate(turn_label):
        if slot1 in data[0]:
            slot_exist_list.append(slot1)
            domain=data[0].split('-')[0]
        if slot2 in data[0]:
            slot_exist_list.append(slot2)

    if slot1 in slot_exist_list and slot2 in slot_exist_list:
        return input_dict
    if slot1 in slot_exist_list:
        if domain not in input_dict:
            input_dict[domain]=[]
        input_dict[domain].append(turn_dialogue)
    return input_dict
def state_process(state_list):
    state={}
    for slot in state_list:
        state[slot['slots'][0][0]]=slot['slots'][0][1]
    return state
def json_read(path):
    with open(path, "r") as f:
        row_data = json.load(f)
    return row_data
def data_process(args, path_name, SLOTS, tokenizer, description,slot_info,ontology, dataset=None,tagging=False):
    slot_lang_list = ["description_human", "rule_description", "value_description", "rule2", "rule3"]
    value_map_dict={}
    if tagging:
        model_path='model/Slot_model/assistant_Slot_Trained_model_bert_classifier_128_0.0001'
        # config=BertConfig.from_pretrained(os.path.join(model_path,'config.json'))
        id2label=json_read(os.path.join(model_path,'config.json'))['id2label']
        crf_model= BertSlotFilling.from_pretrained(model_path)
        crf_tokenizer=BertTokenizer.from_pretrained(os.path.join(model_path,'vocab.txt'))
        # crf_model=torch.load(os.path.join(model_path,'best_slot_model.bin'))
        # transformers_model = BertForTokenClassification(config)
        # transformers_model.bert = BertModel.from_pretrained("bert-base-uncased")
        # crf_model = BertSlotFilling(transformers_model, {i: label for i, label in enumerate(id2label.values())})
        # crf_model.load_state_dict(torch.load(os.path.join(model_path, 'pytorch_model.bin')), strict=False)
    
    print(("Reading all files from {}".format(path_name)))
    data = []
    domain_counter = {}
    value_type_counter={}
    slot_index={}
    for idx,slot in enumerate(description.keys()):
         slot_index[slot]=idx
    index_slot={v: k for k, v in slot_index.items()}
    # read files
    departure_dict={}
    destination_dict={}
    arriveby_dict={}
    leaveat_dict={}
    return_confusion_dict={}
    
    with open(path_name) as f:
        dials = json.load(f)
        # print('dials',dials[0])

        if dataset=="train" and args["fewshot"]>0:
            dials = finetune_data(args, path_name)
        dials_list=[]
        for dial_dict in tqdm(dials):
            dialog_history = ""
            if 'hospital' in dial_dict["domains"] or 'police' in dial_dict["domains"]:
                # print(dial_dict["domains"])

                continue
            # Counting domains
            for domain in dial_dict["domains"]:
                if domain not in EXPERIMENT_DOMAINS:
                    continue
                if domain not in domain_counter.keys():
                    domain_counter[domain] = 0
                domain_counter[domain] += 1

            # Unseen domain setting
            if args["only_domain"] != "none" and args["only_domain"] not in dial_dict["domains"]:
                continue
            if (args["except_domain"] != "none" and dataset == "test" and args["except_domain"] not in dial_dict["domains"]) or \
            (args["except_domain"] != "none" and dataset != "test" and [args["except_domain"]] == dial_dict["domains"]):
                continue

            dialogue={}
            for key in dial_dict.keys():
                if key!='dialogue':
                   dialogue[key]=dial_dict[key]
            # Reading data
            # turn_slot_values = []
            # final_state=dial_dict["turns"][-1]["state"]["slot_values"]
            # print(' final_state', final_state)
            candiate_slot=possible_slot_values(args,dial_dict,SLOTS,dataset)
            # print('candiate_slot',candiate_slot)
            if args['use_2.4']:
                turns=enumerate(dial_dict["dialogue"])
            else:
                turns= enumerate(dial_dict["turns"])
            dialog_history={}
            history_slot_type={}
            # init_input_state={k:'none' for k in description.keys()}
            init_input_state = {'empty-state':'null'}
            candiate_dict={}
            turn_list=[]
            for ti, turn in turns:
                # starttime = datetime.datetime.now()
                turn_fix={}
                turn_id = ti

                if args['use_2.4']:
                    if turn["system_transcript"]=="":
                        turn["system_transcript"]='none'
                    turn_dialogue= ("System: " + turn["system_transcript"] +" User: " + turn["transcript"])
                    dialog_history[ti]= [turn["system_transcript"],turn["transcript"]]
                    turn['belief_state'] = state_filter(turn['belief_state'])
                    state_dict=state_process(turn['belief_state'])
                    # print(turn['belief_state'])
                else:
                    turn_dialogue=(" System: " + turn["system"] + " User: " + turn["user"]) if turn["system"]!= "" else (" System: " + '[none]' + " User: " + turn["user"])
                # accumulate dialogue utterances
                    dialog_history[ti]=( (" System: " + turn["system"] + " User: " + turn["user"]) if turn["system"]!='none' else (" System: " + '[none]' + " User: " + turn["user"]) )
                if args["fix_label"]:
                    slot_values = fix_general_label_error(turn["state"]["slot_values"],SLOTS)
                else:
                    if args['use_2.4']:
                        slot_values = state_process_24(turn["belief_state"])
                    else:
                        slot_values = turn["state"]["slot_values"]

                departure_dict=confusion_slot_collect(input_dict=departure_dict, slot1='departure', slot2='destination', turn_label=turn['turn_label'], turn_dialogue=dialog_history[ti])
                destination_dict=confusion_slot_collect(input_dict=destination_dict, slot1='destination', slot2='departure', turn_label=turn['turn_label'], turn_dialogue=dialog_history[ti])
                arriveby_dict =  confusion_slot_collect(input_dict=arriveby_dict, slot1='arriveby', slot2='leaveat', turn_label=turn['turn_label'], turn_dialogue=dialog_history[ti])
                leaveat_dict = confusion_slot_collect(input_dict=leaveat_dict, slot1='leaveat', slot2='arriveby', turn_label=turn['turn_label'], turn_dialogue=dialog_history[ti])
                # print('departure_dict',departure_dict.keys())
                # Generate domain-dependent slot list
                if not args['pre_train_data']:
                    slot_temp = SLOTS
                    if dataset == "train" or dataset == "dev":
                        if args["except_domain"] != "none":
                            slot_temp = [k for k in SLOTS if args["except_domain"] not in k]
                            slot_values = OrderedDict([(k, v) for k, v in slot_values.items() if args["except_domain"] not in k])
                        elif args["only_domain"] != "none":
                            slot_temp = [k for k in SLOTS if args["only_domain"] in k]
                            slot_values = OrderedDict([(k, v) for k, v in slot_values.items() if args["only_domain"] in k])
                    else:
                        if args["except_domain"] != "none":
                            slot_temp = [k for k in SLOTS if args["except_domain"] in k]
                            slot_values = OrderedDict([(k, v) for k, v in slot_values.items() if args["except_domain"] in k])
                        elif args["only_domain"] != "none":
                            slot_temp = [k for k in SLOTS if args["only_domain"] in k]
                            slot_values = OrderedDict([(k, v) for k, v in slot_values.items() if args["only_domain"] in k])

                    if ti==0:
                        input_pre_turn_state=init_input_state
                        preturn_type={}
                        preturn_candiate={}
                        # print(init_input_state)
                    else:
                        # input_pre_turn_state=input_pre_turn_state_process(input_pre_turn_state,pre_turn_state)
                        input_pre_turn_state=pre_turn_state

                    turn_belief_list = {str(k):str(v) for k,v in slot_values.items()}
                    if  not tagging:
                        BIO_INIT=[]
        
                        for token in tokenizer.tokenize(turn_dialogue):
                            BIO_INIT.append('O')
                            # print('token',token)
                        # assert len(BIO_INIT)==len(turn_dialogue)
                        slot_operation_dict={}
                        history_slot_type[ti]=[]
                        for slot in candiate_slot.keys():


                            slot_candidate=ontology[slot]

                            value_type=description[slot]['value_type']
                            if value_type == 'bool':
                                slot_candidate.append(slot.split('-')[1])
                            # print(slot_candidate)
                            for value in slot_candidate:
                                    if value in ['none','no','dontcare','yes']:
                                        continue
                                #  if value in turn_dialogue.split():
                                    matching_result=value_matching(value, turn_dialogue.split())
                                    if matching_result['matching_result']:
                                    #    print('value',value,turn_dialogue.split())
                            # if final_state[slot] in turn_dialogue:
                                # print(slot,final_state[slot],description[slot]['value_type'])
                                        if description[slot]['value_type'] not in history_slot_type[ti]:
                                           history_slot_type[ti].append(description[slot]['value_type'])
                                        if description[slot]['value_type'] not in candiate_dict:
                                            candiate_dict[description[slot]['value_type']]=[]
                                        if value not in candiate_dict[description[slot]['value_type']]:

                                            candiate_dict[description[slot]['value_type']].append(value)
                                        # if value_type!='bool':
                                        if matching_result['matching_value'] not in value_map_dict:
                                            value_map_dict[matching_result['matching_value']]=value
                                        BIO_INIT=BIO_tagging(BIO_INIT,turn_dialogue,matching_result['matching_value'],value_type)

                        for slot in slot_list:
                            slot_operation_dict[slot]=slot_operation(slot,{},slot_values,state_dict) if ti==0 else slot_operation(slot,pre_turn_state,slot_values,state_dict)

                        BIO_STR=list_to_str(BIO_INIT)
                        if args['make_tagging_label']:



                            turn['input'] = turn_dialogue
                            turn["BIO_TAGGING"] = BIO_STR
                            # endtime = datetime.datetime.now()
                            # print('timemiddle', (endtime - starttime))

                        else:
                            turn['input'] = turn_dialogue
                            turn["BIO_TAGGING"] = BIO_STR
                            turn['input'] = turn_dialogue
                            turn["BIO_TAGGING"] = BIO_STR
                            turn['operation'] = slot_operation_dict.copy()
                            turn['dialog_history'] = dialog_history.copy()
                            turn['history_slot_type'] = value_type_process(history_slot_type.copy())
                            turn['input_pre_turn_state'] = input_pre_turn_state.copy()
                            turn['candiate_dict'] = turn_label_check(turn['turn_label'], candiate_dict.copy(), description)
                            turn['turn_label'] = turn_label_process(turn['turn_label'])
                            turn['turn_id'] = dial_dict['dialogue_idx']
                            turn['domains'] = dial_dict['domains']
                            turn['input_history'] = history_concat(turn["dialog_history"], tokenizer)
                            turn['att_pre_turn_state'] = att_pre_turn_state_process(turn["input_pre_turn_state"], SLOTS)
                            # turn['star_history'],turn['star_utter']=star_input_data(turn['dialog_history'],turn['input_pre_turn_state'])
                            turn_fix["system_transcript"] = turn["system_transcript"]
                            turn_fix["turn_idx"] = turn["turn_idx"]
                            turn_fix["turn_label"] = turn["turn_label"]
                            turn_fix["transcript"] = turn["transcript"]
                            turn_fix["domain"] = turn["domain"]
                            turn_fix["operation"] = turn["operation"]
                            turn_fix["dialog_history"] = turn["dialog_history"]
                            turn_fix["history_slot_type"] = turn["history_slot_type"]
                            turn_fix["input_pre_turn_state"] = turn["input_pre_turn_state"]
                            turn_fix["candiate_dict"] = turn["candiate_dict"]
                            turn_fix["turn_id"] = turn["turn_id"]
                            turn_fix['input_history'] = history_concat(turn["dialog_history"], tokenizer)
                            turn_fix['att_pre_turn_state'] = att_pre_turn_state_process(turn["input_pre_turn_state"], SLOTS)
                    elif  dataset=='test' and tagging:
                        # print('hhh')
                        BIO_result=tagging_model(crf_model,turn_dialogue,crf_tokenizer,id2label)
                        print('BIO_result',BIO_result,turn_dialogue)
                        turn['history_slot_type']=test_value_type_process(preturn_type,BIO_result,ti).copy()
                        preturn_type=turn['history_slot_type']
                        turn['candiate_dict']=test_value_can_process(preturn_candiate,BIO_result).copy()
                        preturn_candiate= turn['candiate_dict']

                    turn_list.append(turn)
                    # pre_turn_list=turn_belief_list
                    pre_turn_state=slot_values
            dialogue['dialogue']=turn_list
            dials_list.append(dialogue)
    # print(len(data))
    print("domain_counter", domain_counter)
    print('value_type_counter',value_type_counter)
    # for detail in data :detail
    #     print('history',['dialog_history'])
    #     print('input',detail['intput_text'])
    if args['use_2.4']:
        if tagging:
         with open(f'{dataset}_dlc_2.4_tagging.json', 'w') as f:
            json.dump(dials_list, f,indent=4)
        elif args['make_tagging_label']:
            with open(f'{dataset}_dlc_2.4_tagging_label.json', 'w') as f:
                json.dump(dials_list, f, indent=4)
            with open(f'{dataset}_dlc_2.4_map_label.json', 'w') as f:
                json.dump(value_map_dict, f, indent=4)
        else:
         with open(f'{dataset}_dlc_2.4.json', 'w') as f:
            json.dump(dials_list, f,indent=4)

    # else:
    #     with open(f'{dataset}_dlc.json', 'w') as f:
    #        json.dump(dials, f,indent=4)
    #     with open(f'{dataset}_co_reference.json', 'w') as f:
    #        json.dump(co_reference_dict, f,indent=4)
    if args['pre_train_data']:
        return_confusion_dict['departure_dict']=departure_dict
        return_confusion_dict['destination_dict'] = destination_dict
        return_confusion_dict['arriveby_dict'] = arriveby_dict
        return_confusion_dict['leaveat_dict'] = leaveat_dict
        with open(f'{dataset}_confusion.json', 'w') as f:
           json.dump(return_confusion_dict, f,indent=4)
def test_value_type_process(preturn_type,tagging,turn_id):

    for key in tagging.keys():
        if key not in preturn_type:
            preturn_type[key]=[]
        preturn_type[key].append(turn_id)
    return    preturn_type
def test_value_can_process(preturn_can,tagging):
    for key in tagging.keys():
        if key not in preturn_can:
            preturn_can[key]=[]
        for value in tagging[key]:
            if value not in preturn_can[key]:
                  preturn_can[key].append(value)
    return preturn_can
def tagging_model(model,turn_dialogue,tokenizer,id2label):
    model.eval()
    model.to(device)
    # input=tokenizer(turn_dialogue,pad_to_max_length=True,
    #                         return_tensors="pt", verbose=False, add_special_tokens=True)
    input= tokenizer(
    turn_dialogue, add_special_tokens=True, return_tensors="pt")
    input_ids=input['input_ids']
    attention_mask=input["attention_mask"]
    token_type_ids=input["token_type_ids"]
    with torch.no_grad():
       predicted_tags = model(input_ids=input_ids.to(device), token_type_ids=token_type_ids.to(device),
                             attention_mask=attention_mask.to(device))
       # print('predicted_tags', turn_dialogue,predicted_tags,tokenizer.decode())
       result = [[id2label[str(data)] for data in datas] for datas in predicted_tags]
       # print('tagging_resut',result)
       tagging_result=value_extract(result[0],tokenizer.tokenize(turn_dialogue))
       return tagging_result
def value_extract(tagging,text):
    result={}
    # print('text',text)
    for idx,token in enumerate(tagging):
        if token!='O':
            # print(token)
            if token[0]=='B':
                if token not in result:
                    result[token[2:]]=[]
                result[token[2:]].append([])
                result[token[2:]][-1].append(text[idx])
            elif token[0]=='I':
                result[token[2:]][-1].append(text[idx])

    result=tagging_result_check(result)
    # print('result', result)
    # for key in result.keys():
    #     key_list=result[key]
    #     for value in key_list:
    #         print('ha',tokenizer.decode(value))
    return result
def tagging_result_check(tagging_result):
        for key in tagging_result.keys():
            key_list=tagging_result[key]
            for idx,value in enumerate(key_list):
                if key=='time':
                    time=''
                    for token in value:
                        time+=token
                    key_list[idx]=time
                else:
                    str = ''
                    for token_idx,token in enumerate(value):
                        if '##' in token:

                               str+=token[2:]

                        else:
                            if  token_idx !=0:
                                str += ' ' +token
                            else:
                                str += token
                    key_list[idx]=str
        return tagging_result
def value_type_process(input_dict):
    result={}
    # print('input_dict',input_dict)
    for turn in input_dict.keys():
        for value_type in input_dict[turn]:
            if value_type not in result.keys():
                result[value_type]=[]
            result[value_type].append(turn) 
    # print('result',result)
    return result
def possible_slot_values(args,dial_dict,SLOTS,dataset):
    result={}
    if args['use_2.4']:
                turns=enumerate(dial_dict["dialogue"])
    else:
                turns= enumerate(dial_dict["turns"])
    for ti, turn in turns:
        
        if args["fix_label"]:
                    slot_values = fix_general_label_error(turn["state"]["slot_values"],SLOTS)
        else:
                    if args['use_2.4']:
                         slot_values = state_process_24(turn["belief_state"])
                        #  print('slot_values',slot_values) 
                    else:
                         slot_values = turn["state"]["slot_values"]
        if dataset == "train" or dataset == "dev":
            if args["except_domain"] != "none":
                
                slot_values = OrderedDict([(k, v) for k, v in slot_values.items() if args["except_domain"] not in k])
            elif args["only_domain"] != "none":
                
                slot_values = OrderedDict([(k, v) for k, v in slot_values.items() if args["only_domain"] in k])
        else:
            if args["except_domain"] != "none":
               
                slot_values = OrderedDict([(k, v) for k, v in slot_values.items() if args["except_domain"] in k])
            elif args["only_domain"] != "none":
             
                slot_values = OrderedDict([(k, v) for k, v in slot_values.items() if args["only_domain"] in k])
        for  slot in slot_values.keys():
            slot_domain=slot[:slot.find('-')]
            if slot_domain not in EXPERIMENT_DOMAINS:
                continue
            if slot not in result:  
               result[slot]=[]
            else:
            #    for value in slot_values[slot]:
                 if slot_values[slot] not in result[slot]:
                    result[slot].append(slot_values[slot])
    return result
def list_to_str(input_list):
    result_str=""
    for idx,token in enumerate(input_list):
           result_str+=token
           if idx!=len(input_list)-1:
              result_str+=" "
    return result_str
def BIO_tagging(BIO_INIT,dialogue,value,value_type):
    Init = 0
    # print('turn_dialogue',dialogue.split())
    # print('value',value)
    tokenizer_dialogue=tokenizer.tokenize(dialogue)
    # print('value',tokenizer_dialogue)
    tokenizer_result=tokenizer(value)
    # if len(value.split())!=len(tokenizer_result['input_ids'])-2:
    #     print('value',value,tokenizer_result)
    value=tokenizer.tokenize(value)
    # print('value',value)
    # print('value',value)
    # dialogue=tokenizer.tokenize(dialogue)
    turn_dialogue=enumerate(tokenizer_dialogue)
    for idx,token in turn_dialogue:
        if token==value[0]:
           for value_idx in range(len(tokenizer_result['input_ids'])-2):
              if value_idx==0:
                 BIO_INIT[value_idx+idx]="B"+"-"+value_type
              else:
                 try:
                   if dialogue[value_idx+idx]==value[value_idx]:
                     BIO_INIT[value_idx+idx]="I"+"-"+value_type
                 except:
                    break
           for skip in range(len(value)):
               try:
                   next(turn_dialogue)
               except:
                    break
    # print('BIO_INIT',BIO_INIT)
    return BIO_INIT
def slot_operation(slot,pre_turn_state,turn_state,state):
    if slot in turn_state.keys():
        if slot not in pre_turn_state.keys():
            return 'update'
        else:
            if turn_state[slot]!=pre_turn_state[slot]:
                return 'update'
            else:
                return  'keep'
    else:
         if slot in pre_turn_state.keys(): 
                 return 'delete' 
         else:
             if slot in state:
                    return  'keep'
             else:
                    return 'dontcare'

def get_slot_information(ontology):
    ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])
    SLOTS = [k.replace(" ","").lower() if ("book" not in k) else k.lower() for k in ontology_domains.keys()]

    return SLOTS

def turn_label_check(turnlabel,candidate_dict,description):
    for data in turnlabel:
        slot=data[0]
        if slot.split("-")[0] not in EXPERIMENT_DOMAINS:
            continue
        value=data[1]
        value_type=description[slot]['value_type']
        if value_type not in candidate_dict:
            candidate_dict[value_type]=[]
        if value not in candidate_dict[value_type]:
            candidate_dict[value_type].append(value)
        if 'none' not in candidate_dict[value_type]:
            candidate_dict[value_type].append('none')
    return candidate_dict
def gpt_collate_fn(data,tokenizer):
    batch_data = {}
    for key in data[0]:
        batch_data[key] = [d[key] for d in data]

    output_batch = tokenizer(batch_data["output_text"], padding=True, return_tensors="pt", add_special_tokens=False, return_attention_mask=False, truncation=True, max_length=1000)
    batch_data["input_ids"] = output_batch['input_ids']
    # print(tokenizer.decode(input_batch["input_ids"]))
    return batch_data

def crf_collate_fn(data, tokenizer):
    batch_data = {}

    for key in data[0]:
        batch_data[key] = [d[key] for d in data]

    max_length = max(len(each.split()) for each in batch_data["input"])
    input_batch=tokenizer.batch_encode_plus(batch_data["input"],
                                add_special_tokens=True,
                                padding='longest',
                                return_tensors="pt")
    batch_data["encoder_input"] = input_batch["input_ids"]
    batch_data["attention_mask"] = input_batch["attention_mask"]
    batch_data["token_type"] = input_batch["token_type_ids"]

    return batch_data
def collate_fn(data, tokenizer):
    batch_data = {}

    for key in data[0]:
        batch_data[key] = [d[key] for d in data]

    # print(batch_data['input_pre_turn_state'])
    for batch_idx in range(len(batch_data["transcript"])):
        batch_data["transcript"][batch_idx]+=" "+tokenizer.sep_token+" "+state_dict_to_str(batch_data['input_pre_turn_state'][batch_idx])
    # print(batch_data["transcript"])
    input_batch = tokenizer(batch_data["system_transcript"],batch_data["transcript"], padding=True, return_tensors="pt", verbose=False, add_special_tokens=True)
    # history_input_batch=tokenizer(batch_data["system_transcript"],batch_data["transcript"], padding=True, return_tensors="pt", verbose=False, add_special_tokens=True)
    batch_data["encoder_input"] = input_batch["input_ids"]
    batch_data["attention_mask"] = input_batch["attention_mask"]
    batch_data["token_type"] = input_batch["token_type_ids"]
    batch_data['input_pre_turn_state'] = batch_state_process(batch_data["input_pre_turn_state"].copy(), tokenizer)
    batch_data['input_history']= batch_history_process_concate(batch_data["input_history"].copy(),tokenizer)
    batch_data['input_att_state']=batch_att_state_process(batch_data['att_pre_turn_state'],tokenizer)
    batch_data['star_input'] = batch_star_input(batch_data['dialog_history'],batch_data['star_utter'],batch_data['input_pre_turn_state'],tokenizer)
    return batch_data
def batch_star_input(history,batch_state,tokenizer,max_seq_length=512):
    result = {}

    result["encoder_input"] = []
    result["attention_mask"] = []
    result["token_type"] = []
    batch_size=len(history)

    for idx in range(batch_size):

        diag_history,diag_utter,state=star_input_data(history[idx],batch_state[idx])
        result["encoder_input"].append([])
        result["attention_mask"].append([])
        result["token_type"].append([])

        # print('batch_data', batch_data)
        avail_length_1 = max_seq_length - len(state) - 3
        diag_1 = tokenizer.tokenize(diag_utter)
        diag_2 = tokenizer.tokenize(diag_history)
        state = tokenizer.tokenize(state)
        avail_length = avail_length_1 - len(diag_1)

        if avail_length <= 0:
            diag_2 = []
        elif len(diag_2) > avail_length:  # truncated
            avail_length = len(diag_2) - avail_length
            diag_2 = diag_2[avail_length:]

        if len(diag_2) == 0 and len(diag_1) > avail_length_1:
            avail_length = len(diag_1) - avail_length_1
            diag_1 = diag_1[avail_length:]

        # we keep the order
        drop_mask = [0] + [1] * len(diag_2) + [0] * len(state) + [0] + [1] * len(diag_1) + [0]  # word dropout

        diag_2 = ["[CLS]"] + diag_2 + state + ["[SEP]"]
        # print('diag_2 ', diag_2)
        diag_1 = diag_1 + ["[SEP]"]
        diag = diag_2 + diag_1
        input_data = tokenizer(diag, padding='max_length',
                                return_tensors="pt", verbose=False, add_special_tokens=True, max_length=512,truncation=True)

        # print('turn',tokenizer.decode(turn_input["input_ids"]))
        result["encoder_input"][idx] = input_data["input_ids"]
        result["attention_mask"][idx] = input_data["attention_mask"]
        result["token_type"][idx] = torch.tensor(drop_mask)

    # print('result["encoder_input"][idx]',result["encoder_input"][idx].size())
    result["encoder_input"] = torch.stack(result["encoder_input"], 0).squeeze(dim=1)
    result["attention_mask"] = torch.stack(result["attention_mask"], 0).squeeze(dim=1)
    result["token_type"] = torch.stack(result["token_type"], 0).squeeze(dim=1)
    # print(result["encoder_input"].size())
    return result
def star_input_data(history_data,pre_state):
    max_history=20
    history=""
    utter=""
    state=""
    for idx in range(len(history_data)):
        if int(idx)!=len(history_data)-1 and idx>=len(history_data)-max_history:
            history += history_data[idx][1]+" "
            if history_data[idx][0]!='none':
                history+=history_data[idx][0]+" "
        elif int(idx)==len(history_data)-1:
            # print(len(history_data))
            utter += history_data[idx][1]+" "
            if history_data[idx][0] != 'none':
                utter += history_data[idx][0]+" "
    for slot in pre_state.keys():
        if slot!='empty-state':
            domain=slot.split('-')[0]
            name = slot.split('-')[1]
            state+=domain+' '+name+' '+pre_state[slot]+' '
    return history,utter,state
def state_dict_to_str(input_dict):
    result=''
    for key in input_dict.keys():
        if key!='empty_state':
           key_list=key.split('-')
           result+=key_list[0]+' '+key_list[1]+' '+input_dict[key]+' '
        else:
            result += key + ' ' + input_dict[key] + ' '
    return result
def collate_test_fn(data, tokenizer):
    batch_data = {}

    for key in data[0]:
        batch_data[key] = [d[key] for d in data]
    for batch_idx in range(len(batch_data["transcript"])):
        batch_data["transcript"][batch_idx]+=" "+tokenizer.sep_token+" "+state_dict_to_str(batch_data['input_pre_turn_state'][batch_idx])
    # print(batch_data["transcript"])
    input_batch = tokenizer(batch_data["system_transcript"],batch_data["transcript"], padding=True, return_tensors="pt", verbose=False, add_special_tokens=True)
    # history_input_batch=tokenizer(batch_data["system_transcript"],batch_data["transcript"], padding=True, return_tensors="pt", verbose=False, add_special_tokens=True)
    batch_data["encoder_input"] = input_batch["input_ids"]
    batch_data["attention_mask"] = input_batch["attention_mask"]
    batch_data["token_type"] = input_batch["token_type_ids"]
    batch_data['pre_turn_state'] = batch_data["input_pre_turn_state"].copy()
    batch_data['input_pre_turn_state']=batch_state_process(batch_data["input_pre_turn_state"],tokenizer)
    batch_data['input_history']= batch_history_process_concate(batch_data["input_history"],tokenizer)
    batch_data['input_att_state']=batch_att_state_process(batch_data['att_pre_turn_state'],tokenizer)
    return batch_data
def collate_pre_fn(data,tokenizer):
    batch_data = {}
    # print(data[0].keys())
    for key in data[0]:
        # print('key',key)
        batch_data[key] = [d[key] for d in data]
        # print(key,len(batch_data[key]))
        batch_data[key] = pre_train_data_process(batch_data[key], tokenizer)


    return batch_data

def pre_train_data_process(input_list,tokenizer):
    result={}
    result["encoder_input"] = []
    result["attention_mask"] = []
    result["token_type"] = []
    for data in input_list:

        # print('data',len(data))
        data_encoder=tokenizer(data, padding='max_length',
                               return_tensors="pt", verbose=False, add_special_tokens=True, max_length=100)

        result["encoder_input"].append(data_encoder["input_ids"])
        result["attention_mask"].append(data_encoder["attention_mask"])
        result["token_type"].append(data_encoder["token_type_ids"])
    # print('result["token_type"]',result["token_type"])
    return result

def batch_att_state_process(batch_pre_state,tokenizer):
    result = {}

    result["encoder_input"] = []
    result["attention_mask"] = []
    result["token_type"] = []
    for idx, batch_data in enumerate(batch_pre_state):
        result["encoder_input"].append([])
        result["attention_mask"].append([])
        result["token_type"].append([])
      
        # print('batch_data', batch_data)
        
        for slot_idx,slot in enumerate(batch_data.keys()):
            result["encoder_input"][idx].append([])
            result["attention_mask"][idx].append([])
            result["token_type"][idx].append([])
            domain=slot.split('-')[0]
            name=slot.split('-')[1]
            input=domain+' '+name+' '+batch_data[slot]
            # print('input',input)
            slot_values=tokenizer(input, padding='max_length',
                               return_tensors="pt", verbose=False, add_special_tokens=True, max_length=20)
            result["encoder_input"][idx][slot_idx]=slot_values["input_ids"]
            result["attention_mask"][idx][slot_idx]=slot_values["attention_mask"]
            result["token_type"][idx][slot_idx]=slot_values["token_type_ids"]
            # print(slot_values["input_ids"].size())
        # print('turn',tokenizer.decode(turn_input["input_ids"]))
        result["encoder_input"][idx]=torch.stack(result["encoder_input"][idx],dim=0).squeeze(dim=1)
        result["attention_mask"][idx]=torch.stack(result["attention_mask"][idx],dim=0).squeeze(dim=1)
        result["token_type"][idx]=torch.stack(result["token_type"][idx],dim=0).squeeze(dim=1)

    # print('result["encoder_input"][idx]',result["encoder_input"][idx].size())
    result["encoder_input"]= torch.stack(result["encoder_input"], 0).squeeze(dim=1)
    result["attention_mask"]= torch.stack(result["attention_mask"], 0).squeeze(dim=1)
    result["token_type"]= torch.stack(result["token_type"], 0).squeeze(dim=1)
    return result
    # print('att_result', result["encoder_input"].size())
def batch_state_process(batch_pre_state,tokenizer):
    result = {}

    result["encoder_input"] = []
    result["attention_mask"] = []
    result["token_type"] = []
    for idx, batch_data in enumerate(batch_pre_state):
        result["encoder_input"].append([])
        result["attention_mask"].append([])
        result["token_type"].append([])
        # result["encoder_input"][idx].append([])
        # result["attention_mask"][idx].append([])
        # result["token_type"][idx].append([])
        # print('batch_data', batch_data)
        input=''
        for key in batch_data.keys():
            # print(key,batch_data[key])
            input+= tokenizer.sep_token+' '+key+':'+' '+batch_data[key]+' '
        # print('input',input)
        turn_input = tokenizer(input, padding='max_length',
                               return_tensors="pt", verbose=False, add_special_tokens=True, max_length=300)
        # print('turn',tokenizer.decode(turn_input["input_ids"]))

        result["encoder_input"][idx]= turn_input["input_ids"]
        result["attention_mask"][idx]= turn_input["attention_mask"]
        result["token_type"][idx] = turn_input["token_type_ids"]

        # print('result["encoder_input"][idx][turn]', result["encoder_input"][idx][turn].size())
    result["encoder_input"]= torch.squeeze(torch.stack(result["encoder_input"], 0), dim=1)
    result["attention_mask"]= torch.squeeze(torch.stack(result["attention_mask"], 0), dim=1)
    result["token_type"]= torch.squeeze(torch.stack(result["token_type"], 0), dim=1)

    # print('result', result["encoder_input"].size())
    return result
def batch_history_process_combine(batch_history,tokenizer):
    result={}
    result["encoder_input"]=[]
    result["attention_mask"] = []
    result["token_type"]=[]
    for idx,batch_data in enumerate(batch_history):
        result["encoder_input"].append([])
        result["attention_mask"].append([])
        result["token_type"].append([])

        # print('batch_data', batch_data)
        for turn in range(len(batch_data)):
            result["encoder_input"][idx].append([])
            result["attention_mask"][idx].append([])
            result["token_type"][idx].append([])
            # result[idx].append({})
            turn_input= tokenizer(batch_data[str(turn)][0],batch_data[str(turn)][1], padding='max_length', return_tensors="pt", verbose=False, add_special_tokens=True,max_length=100)
            # print('turn',turn,result)

            result["encoder_input"][idx][turn]=turn_input["input_ids"]
            result["attention_mask"][idx][turn]=turn_input["attention_mask"]
            result["token_type"][idx][turn]=turn_input["token_type_ids"]

        # print('result["encoder_input"][idx][turn]', result["encoder_input"][idx][turn].size())
        result["encoder_input"][idx] = torch.squeeze(torch.stack(result["encoder_input"][idx], 0),dim=1)
        result["attention_mask"][idx] = torch.squeeze(torch.stack(result["attention_mask"][idx], 0),dim=1)
        result["token_type"][idx] = torch.squeeze(torch.stack(result["token_type"][idx], 0),dim=1)

        # print('result["encoder_input"][idx]', result["encoder_input"][idx].size())
    return result
def batch_history_process_concate(batch_history,tokenizer):
    result={}

    # for idx,batch_data in enumerate(batch_history):
    #     result["encoder_input"].append([])
    #     result["attention_mask"].append([])
    #     result["token_type"].append([])
    #
    #     # print('batch_data', batch_data)
    #
    #
    #     # result[idx].append({})
    # print('batch_history',batch_history)
    turn_input= tokenizer(batch_history, padding='max_length', return_tensors="pt", verbose=False, add_special_tokens=True,truncation=True,max_length=512)
    #     # print('turn',turn,result)
    #
    result["encoder_input"]=turn_input["input_ids"]
    result["attention_mask"]=turn_input["attention_mask"]
    result["token_type"]=turn_input["token_type_ids"]
    #
    #     # print('result["encoder_input"][idx][turn]', result["encoder_input"][idx][turn].size())
    #     result["encoder_input"][idx] = torch.squeeze(torch.stack(result["encoder_input"][idx], 0),dim=1)
    #     result["attention_mask"][idx] = torch.squeeze(torch.stack(result["attention_mask"][idx], 0),dim=1)
    #     result["token_type"][idx] = torch.squeeze(torch.stack(result["token_type"][idx], 0),dim=1)

        # print('result["encoder_input"][idx]', result["encoder_input"][idx].size())
    return result
def prepare_data(args, tokenizer):
   
    if args['use_2.4']:
       path_train = 'data_2.4/data/mwz2.4/train_dials.json'
       path_dev = 'data_2.4/data/mwz2.4/dev_dials.json'
       path_test = 'data_2.4/data/mwz2.4/test_dials.json'
       if args['tagging']:
             dlc_train = 'train_dlc_2.4_tagging_label.json'
             dlc_dev = 'dev_dlc_2.4_tagging_label.json'
             dlc_test = 'test_dlc_2.4_tagging_label.json'
       else:
           dlc_train='train_dlc_2.4.json'
           dlc_dev='dev_dlc_2.4.json'
           dlc_test='test_dlc_2.4.json'
       ontology = json.load(open("data_2.4/data/mwz2.4/ontology.json", 'r'))
    #    print('ontology',ontology)
    else:
        path_train = 'data/train_dials.json'
        path_dev = 'data/dev_dials.json'
        path_test = 'data/test_dials.json'
        dlc_train='train_dlc.json'
        dlc_dev='dev_dlc.json'
        dlc_test='test_dlc.json'
        ontology = json.load(open("data/ontology.json", 'r'))
    
    ALL_SLOTS = get_slot_information(ontology)
    description = json.load(open("utils/slot_description.json", 'r'))
    slot_temp={}
    for key in description.keys():
        slot_temp[key]=description[key]['slottype']

    slot_index = {}
    slot_type = {}

    for idx, slot in enumerate(description.keys()):
        slot_index[slot] = idx
        slot_type[slot] = description[slot]["value_type"]
    index_slot = {v: k for k, v in slot_index.items()}
    slot_info={}
    slot_info['slot_index']=slot_index
    slot_info['index_slot']=index_slot
    slot_info['slot_type']=slot_type
    slot_info['description']=description
    if args['data_process']:
        data_process(args, path_train, ALL_SLOTS, tokenizer, description, slot_info,ontology,"train")
        data_process(args, path_dev, ALL_SLOTS, tokenizer, description, slot_info,ontology,"dev")
        data_process(args, path_test, ALL_SLOTS, tokenizer, description,slot_info,ontology, "test")
        if not args['data_loader']:
            os._exit()
    if args['tagging']:
        data_process(args, path_test, ALL_SLOTS, tokenizer, description, slot_info, ontology, "test",tagging=True)
        os._exit()
    if args['data_loader']:
        data_train=read_data(args, dlc_train, "train")
        data_dev=read_data(args, dlc_dev, "dev")
        data_test=read_data(args, dlc_test, "test")
        if args['pre_train_data']:
            pretrain_train=read_json_data('train_confusion.json',args)
            pretrain_dev = read_json_data('dev_confusion.json',args)
            pretrain_test = read_json_data('test_confusion.json',args)
            pre_train_dataset = DSTDataset(pretrain_train, args)
            pre_dev_dataset = DSTDataset(pretrain_dev, args)
            pre_test_dataset = DSTDataset(pretrain_test, args)
        train_dataset = DSTDataset(data_train, args)
        dev_dataset = DSTDataset(data_dev, args)
        test_dataset = DSTDataset(data_test, args)


    if args['crf']:
        train_loader = DataLoader(train_dataset, batch_size=args["train_batch_size"], shuffle=True,
                                  collate_fn=partial(crf_collate_fn, tokenizer=tokenizer), num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=args["test_batch_size"], shuffle=False,
                                 collate_fn=partial(crf_collate_fn, tokenizer=tokenizer), num_workers=0)
        dev_loader = DataLoader(dev_dataset, batch_size=args["dev_batch_size"], shuffle=False,
                                collate_fn=partial(crf_collate_fn, tokenizer=tokenizer), num_workers=0)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args["train_batch_size"], shuffle=True, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=1)
        test_loader = DataLoader(test_dataset, batch_size=args["test_batch_size"], shuffle=False, collate_fn=partial(collate_test_fn, tokenizer=tokenizer), num_workers=1)
        dev_loader = DataLoader(dev_dataset, batch_size=args["dev_batch_size"], shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=1)
    if args['pre_train_data']:
        pre_train_loader = DataLoader(pre_train_dataset, batch_size=args["train_batch_size"], shuffle=True,
                                  collate_fn=partial(collate_pre_fn, tokenizer=tokenizer), num_workers=16)
        pre_test_loader = DataLoader(pre_test_dataset, batch_size=args["test_batch_size"], shuffle=False,
                                 collate_fn=partial(collate_pre_fn, tokenizer=tokenizer), num_workers=16)
        pre_dev_loader = DataLoader(pre_dev_dataset, batch_size=args["dev_batch_size"], shuffle=False,
                                collate_fn=partial(collate_pre_fn, tokenizer=tokenizer), num_workers=16)
    # print('train_loader',train_loader)
        return pre_train_loader,  pre_dev_loader, pre_test_loader
    else:
        return train_loader, dev_loader, test_loader, slot_temp, slot_index, slot_type
def history_concat(history_dict,tokenizer):
    history_list=[]
    result=''
    result+=tokenizer.cls_token+' '
    for turn_idx,turn in enumerate(history_dict.keys()):

        if turn_idx!=len(history_dict)-1:
           history_list.append([history_dict[turn][0],history_dict[turn][1]])
    history_list=reversed(history_list)
    for data in history_list:
       result+=data[0]+' '+data[1]+' '+tokenizer.sep_token+' '

    return result
if __name__ == "__main__":
    from config import get_args
    # import pytorch_lightning as pl
    # from pytorch_lightning import Trainer, seed_everything
    # from transformers import (AdamW, T5Tokenizer, BartTokenizer, BartForConditionalGeneration, T5ForConditionalGeneration, WEIGHTS_NAME,CONFIG_NAME)
    from transformers import BertModel, BertForSequenceClassification, BertTokenizer, BertConfig, BertForTokenClassification
    args = get_args()
    args = vars(args)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    args['slot_lang'] = 'human'

    if args['pre_train_data']:
        pre_train_loader, pre_dev_loader, pre_test_loader= prepare_data(args,tokenizer)
    else:
        train_loader, val_loader, test_loader, ALL_SLOTS, fewshot_loader_dev, fewshot_loader_test = prepare_data(args,tokenizer)
