import torch
import numpy as np
import json
import time
import os
from tqdm import tqdm
from copy import deepcopy

from utils.label_lookup import get_label_ids, combine_slot_values


def model_evaluation(model, test_data, tokenizer, slot_meta, label_list, epoch,
                     args, value_lookup, slot_value_pos,
                     is_gt_p_state=False, is_dev=True):
    """
    label_list:test时使用i.candidate_label_list
    value_lookup:dev时用参数value_lookup，test时用_label_ids, _label_type_ids, _label_mask传参数进模型计算value_lookup
    slot_value_pos:test时使用i.slot_value_pos
    """

    model.eval()

    final_count = 0
    loss = 0.
    joint_acc = 0.
    joint_turn_acc = 0.
    final_joint_acc = 0.  # only consider the last turn in each dialogue
    slot_acc = np.array([0.] * len(slot_meta))
    final_slot_acc = np.array([0.] * len(slot_meta))

    description = json.load(open("utils/slot_description.json", 'r'))
    slot_type = [description[slot]["value_type"] for slot in slot_meta]

    results = {}
    last_dialogue_state = {}
    wall_times = []
    for di, i in enumerate(tqdm(test_data)):
        if i.turn_id == 0 or is_gt_p_state:
            last_dialogue_state = deepcopy(i.gold_last_state)

        i.last_dialogue_state = deepcopy(last_dialogue_state)
        i.renew_instance_state(tokenizer)

        input_ids = torch.LongTensor([i.input_id]).to(model.device)
        input_mask = torch.LongTensor([i.input_mask]).to(model.device)
        segment_ids = torch.LongTensor([i.segment_id]).to(model.device)

        if is_dev:
            label_ids = torch.LongTensor([i.label_ids]).to(model.device)
        else: #test
            label_ids = torch.LongTensor([i.candidate_label_ids]).to(model.device)
            label_list = i.candidate_label_list
            slot_value_pos = i.slot_value_pos
        #label_ids = torch.LongTensor([i.label_ids]).to(model.device) #调试用

        input_ids_state = torch.LongTensor([i.input_id_state]).to(model.device)
        input_mask_state = torch.LongTensor([i.input_mask_state]).to(model.device)
        segment_ids_state = torch.LongTensor([i.segment_id_state]).to(model.device)

        input_token_turn_list = np.array(i.input_token_turn_id)
        history_type_turn_id_list = i.history_type_turn_id


        num_labels = [len(labels) for labels in label_list]

        if not is_dev:
            _label_ids = i._label_ids.to(model.device)
            '''
            #调试用
            new_label_list, _ = combine_slot_values(slot_meta, label_list)
            _label_ids, label_lens = get_label_ids(new_label_list, tokenizer)
            _label_ids = _label_ids.to(model.device)
            '''
            _label_type_ids = torch.zeros(_label_ids.size(), dtype=torch.long).to(model.device)
            _label_mask = (_label_ids > 0).to(model.device)

        start = time.perf_counter()
        with torch.no_grad():
            if is_dev:
                t_loss, _, t_acc, t_acc_slot, t_pred_slot = model(input_ids=input_ids,
                                                                  attention_mask=input_mask,
                                                                  token_type_ids=segment_ids,
                                                                  input_ids_state=input_ids_state,
                                                                  input_mask_state=input_mask_state,
                                                                  segment_ids_state=segment_ids_state,
                                                                  labels=label_ids,
                                                                  input_token_turn_list=input_token_turn_list,
                                                                  history_type_turn_id_list=history_type_turn_id_list,
                                                                  slot_type=slot_type,
                                                                  num_labels=num_labels,
                                                                  slot_value_pos=slot_value_pos,
                                                                  value_lookup=value_lookup,
                                                                  eval_type="dev")
            else:
                t_loss, _, t_acc, t_acc_slot, t_pred_slot = model(input_ids=input_ids,
                                                                  attention_mask=input_mask,
                                                                  token_type_ids=segment_ids,
                                                                  input_ids_state = input_ids_state,
                                                                  input_mask_state = input_mask_state,
                                                                  segment_ids_state = segment_ids_state,
                                                                  labels=label_ids,
                                                                  input_token_turn_list=input_token_turn_list,
                                                                  history_type_turn_id_list=history_type_turn_id_list,
                                                                  slot_type=slot_type,
                                                                  num_labels=num_labels,
                                                                  slot_value_pos=slot_value_pos,
                                                                  _label_ids=_label_ids,
                                                                  _label_type_ids=_label_type_ids,
                                                                  _label_mask=_label_mask,
                                                                  eval_type="test")
            loss += t_loss.item()
            joint_acc += t_acc
            slot_acc += t_acc_slot
            if i.is_last_turn:
                final_count += 1
                final_joint_acc += t_acc
                final_slot_acc += t_acc_slot

        end = time.perf_counter()
        wall_times.append(end - start)

        ss = {}
        t_turn_label = []
        for s, slot in enumerate(slot_meta):
            v = label_list[s][t_pred_slot[0, s].item()]
            if v != last_dialogue_state[slot]:
                t_turn_label.append(slot + "-" + v)
            last_dialogue_state[slot] = v
            if is_dev:
                vv = label_list[s][i.label_ids[s]]
            else:
                vv = label_list[s][i.candidate_label_ids[s]]
            if v == vv:
                continue
            # only record wrong slots
            ss[slot] = {}
            ss[slot]["pred"] = v
            ss[slot]["gt"] = vv

        if set(t_turn_label) == set(i.turn_label):
            joint_turn_acc += 1

        key = str(i.dialogue_id) + '_' + str(i.turn_id)
        results[key] = ss

    loss = loss / len(test_data)
    joint_acc_score = joint_acc / len(test_data)
    joint_turn_acc_score = joint_turn_acc / len(test_data)
    slot_acc_score = slot_acc / len(test_data)
    final_joint_acc_score = final_joint_acc / final_count
    final_slot_acc_score = final_slot_acc / final_count

    latency = np.mean(wall_times) * 1000  # ms

    print("------------------------------")
    print('is_gt_p_state: %s' % (str(is_gt_p_state)))
    print("Epoch %d loss : " % epoch, loss)
    print("Epoch %d joint accuracy : " % epoch, joint_acc_score)
    print("Epoch %d joint turn accuracy : " % epoch, joint_turn_acc_score)
    print("Epoch %d slot accuracy : " % epoch, np.mean(slot_acc_score))
    print("Final Joint Accuracy : ", final_joint_acc_score)
    print("Final slot Accuracy : ", np.mean(final_slot_acc_score))
    print("Latency Per Prediction : %f ms" % latency)
    print("-----------------------------\n")

    if not os.path.exists("pred"):
        os.makedirs("pred")
    json.dump(results, open('pred/preds_%d.json' % epoch, 'w'), indent=4)

    scores = {'epoch': epoch, 'loss': loss, 'joint_acc': joint_acc_score, 'joint_turn_acc': joint_turn_acc_score,
              'slot_acc': slot_acc_score, 'ave_slot_acc': np.mean(slot_acc_score),
              'final_joint_acc': final_joint_acc_score, 'final_slot_acc': final_slot_acc_score,
              'final_ave_slot_acc': np.mean(final_slot_acc_score)}

    return scores