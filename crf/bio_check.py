from tqdm import tqdm
import json
label_path='test_dlc_2.1.json'
tagging_path='test_dlc_2.1_tagging.json'
EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]

def read_data(path_name):
    slot_lang_list = ["description_human", "rule_description", "value_description", "rule2", "rule3"]
    co_reference_dict = {}
    print(("Reading all files from {}".format(path_name)))
    data = []
    domain_counter = {}
    value_type_counter = {}
    # read files
    data = []
    with open(path_name) as f:
        dials = json.load(f)
        # print('dials',dials[0])



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

            # if args['use_2.4']:
            turns = enumerate(dial_dict["dialogue"])
            # else:
            #     turns= enumerate(dial_dict["turns"])
            dialog_history = {}
            for ti, turn in turns:
                # candidate_check(turn)
                # assert len(turn['input'].split())==len(turn['BIO_TAGGING'].split())
                # turn['input']=turn['input'].split()
                data.append(turn)
    return data
def json_read(path):
    with open(path, "r") as f:
        row_data = json.load(f)
    return row_data
def miss_label_check(label,tagging):
    result={}
    result['miss_slot']=[]
    result['miss_value']=[]
    for key in label.keys():
        if key not in tagging:
            result['miss_slot'].append(key)
            continue
        for value in label[key]:
            if value not in tagging[key]:
                if value!='none':
                   result['miss_value'].append({key:value})
    return result
label=json_read(label_path)
tagging=json_read(tagging_path)
result=[]
for dial_idx,dial_dict in tqdm(enumerate(label)):
    dial_name=dial_dict['dialogue_idx']
    result.append({})
    result[-1]['dialogue_idx']=dial_name
    result[-1]['turns']=[]
    turns = enumerate(dial_dict["dialogue"])
    for ti, turn in turns:
        turn_list={}
        # print('turn',turn)
        turn_list['label_candiate_dict']=turn['candiate_dict']
        turn_list['predict_candiate_dict']=tagging[dial_idx]['dialogue'][ti]['candiate_dict']
        turn_list['input']=turn['system_transcript']+' '+turn['transcript']
        turn_list['static']=miss_label_check(turn_list['label_candiate_dict'],turn_list['predict_candiate_dict'])
        result[-1]['turns'].append(turn_list)

with open(f'test_dlc_2.1_check_label.json', 'w') as f:
                json.dump(result, f, indent=4)