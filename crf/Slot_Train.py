import os
import matplotlib
matplotlib.use('Agg')
from typing import Tuple, Dict
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import logging
from logger import logger
import pandas as pd
import copy
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from Model_crf import *
from data_prep import read_data
import matplotlib.pyplot as plt
import os
from sklearn.metrics import *


ilogger = logging.getLogger("info")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print("device: {} n_gpu: {}".format(device, n_gpu))

def read_slots(path1, path2):
    data = pd.read_csv(path1, sep=',', header=0)
    texts = data['text']
    labels = data['label']

    data = pd.read_csv(path2, sep=',', header=0)
    slots = data['text']
    labels = data['label']

    return texts, slots
def label_process(label_batch,label2id,max_length):
    # print(label_batch)
    label_batch = pad_slot_labels_to_max_length(label_batch)
    try:
       batch_label_ids = torch.tensor([[label2id[each] for each in label] for label in label_batch])
    except:
        print(label_batch)
        raise ValueError
    # print(batch_label_ids.size())
    return batch_label_ids

def pad_slot_labels_to_max_length(lst):
    max_length = max(len(each.split()) for each in lst)
    # print('max_length',max_length)
    for i in range(len(lst)):
        now = lst[i].split()
        lst[i] = ["O"] + now + ["O"] * (max_length - len(now) + 1)
    return lst
class BertforSlot:

    def train(self, train_batch_data,dev_batch_data, test_batch_data, description, save_dir, num_epoches, train_batch_size, learning_rate,tokenizer ):

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        logging.basicConfig(filename=os.path.join(save_dir, 'log.txt'), level=logging.INFO)

        # query_list, slot_list = read_slots(data_dir + "/dongao_nlu_v1.csv", data_dir + "/dongao_slot_v1.csv")
        # query_list, slot_list = read_data(data_dir + "/data.csv")
        # print('slot_list',slot_list)
        # train_query, train_slot = read_data(data_dir + "/train.csv")
        # dev_query, dev_slot = read_data(data_dir + "/dev.csv")
        # test_query, test_slot = read_data(data_dir + "/test.csv")
        #
        # slot_labels = [out_line.strip().split(" ") for out_line in slot_list if out_line.strip()]
        # train_slot = [out_line.strip().split(" ") for out_line in train_slot if out_line.strip()]
        # dev_slot = [out_line.strip().split(" ") for out_line in dev_slot if out_line.strip()]
        # test_slot = [out_line.strip().split(" ") for out_line in test_slot if out_line.strip()]

        # print(len(train_query), len(train_slot))
        # print(len(dev_query), len(dev_slot))
        # print(len(test_query), len(test_slot))
        # print('slot_labels',slot_labels)
        all_slot_keys = []
        # print('slot_labels',slot_labels)
        for slot in description:
            # if description[slot]['value_type']=='bool':
            #     continue
            if description[slot]['value_type'] not in all_slot_keys:
                all_slot_keys.append(description[slot]['value_type'])

        # for k in range(len(slot_labels)):
        #     # print(slot_labels[k])
        #     for i in range(len(slot_labels[k])):
        #         label = slot_labels[k][i]
        #         # print('label',label)
        #         if label[0] == "B":
        #             all_slot_keys.append(label[2:])

        all_slot_keys = list(set(all_slot_keys))
        print('all_slot_keys:', all_slot_keys)

        # 标签信息
        labels = ["O"]
        for slot_key in all_slot_keys:
            labels.append("B-" + slot_key)
            labels.append("I-" + slot_key)
        print('labels',labels)
        label2id = {label: i for i, label in enumerate(labels)}
        id2label = {str(i): label for i, label in enumerate(labels)}
        num_labels = len(labels)
        print(label2id)
        print(id2label)

        # 载入bert预训练配置
        config = BertConfig.from_pretrained('bert-base-uncased')
        # print('config',config)
        config.num_labels = num_labels
        config.label2id = label2id
        config.id2label = id2label
        # print('num_labels',config.num_labels)
        # print('config',config)
        # 获取bert预训练词表
        tokenizer =BertTokenizer.from_pretrained('bert-base-uncased')

        # 准备分类模型、载入bert预训练模型
        transformers_model = BertForTokenClassification(config)
        transformers_model.bert = BertModel.from_pretrained("bert-base-uncased")

        model = BertSlotFilling(transformers_model, {i: label for i, label in enumerate(labels)})
        # model_path = './model/Slot_model/assistant_Slot_Trained_model_bert_classifier_128_5e-05'
        # model = BertSlotFilling.from_pretrained(model_path)
        # tokenizer = BertTokenizer.from_pretrained(os.path.join(model_path, 'vocab.txt'))
        # print(model)

        model.to(device)
        logger.info("Successfully loaded pretrained BERT model")

        # 最大长度
        # def pad_slot_labels_to_max_length(lst):
        #     max_length = max(len(each) for each in lst)
        #     for i in range(len(lst)):
        #         now = lst[i]
        #         lst[i] = ["O"] + now + ["O"] * (max_length - len(now) + 1)
        #     return lst



        logger.info("Successfully converted dataset to tensors")

        # 执行训练
        logger.info("Start optimization")
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        global_step = 0
        best_score = 0
        best_model = None
        wait = 0
        wait_patient = 5
        eval_loss_list = []
        epoch_list = []
        for epoch in range(num_epoches):
            loss_sum = 0.0
            acc_sum = 0
            total_sum = 0
            model.train()
            ##
            train_start_time = time.time()
            for train_data in tqdm(train_batch_data):
                input_ids = train_data['encoder_input'].cuda()
                token_type_ids = train_data['token_type'].cuda()
                attention_mask = train_data['attention_mask'].cuda()
                # print('input_ids',input_ids.size())
                labels=label_process(train_data['BIO_TAGGING'],label2id,max_length=input_ids.size()[1]).cuda()
                # print(input_ids.size(),labels.size())

                optimizer.zero_grad()
                loss, logits = model(input_ids=input_ids, token_type_ids=token_type_ids,
                                     attention_mask=attention_mask, labels=labels)
                print('logits',logits.size(),labels.size())
                # [128, 39, 31]
                eval_loss_list.append(loss.item())
                loss_sum += loss.item()
                total_sum += attention_mask[:, 1:-1].reshape(-1).int().sum().item()
                acc_sum += (attention_mask[:, 1:-1].reshape(-1).int() *
                            ((torch.argmax(logits[:, 1:-1, :].reshape(-1, num_labels), dim=-1) ==
                              labels[:, 1:-1].reshape(-1)).int())).sum().item()
                loss.backward()
                optimizer.step()
                global_step += 1
                epoch_list.append(global_step)
            train_end_time = time.time()
            train_time = train_end_time - train_start_time
            step_time = train_time / len(train_batch_data)
            train_loss = loss_sum / len(train_batch_data)
            train_acc = acc_sum / total_sum
            print(f"Training info: epoch {epoch}, step {global_step}, "
                  f"train_loss {train_loss:.6f}, train_acc {train_acc:.6f}, "
                  f"train_time: {train_time:.2f}s ({step_time:.2f}s per step), ")

            ##
            eval_acc_sum = 0.0
            eval_loss_sum = 0.0
            eval_start_time = time.time()
            for val_data in tqdm(dev_batch_data):
                input_ids = val_data['encoder_input'].cuda()
                token_type_ids = val_data['token_type'].cuda()
                attention_mask = val_data['attention_mask'].cuda()
                labels=label_process(val_data['BIO_TAGGING'],label2id,max_length=input_ids.size()[1]).cuda()

                eval_loss, eval_acc= self.eval(model=model,
                                            input_ids=input_ids,
                                            token_type_ids=token_type_ids,
                                            attention_mask=attention_mask,
                                            labels=labels)

                eval_acc_sum += eval_acc
                eval_loss_sum += eval_loss
            eval_end_time = time.time()
            eval_time = eval_end_time - eval_start_time
            eval_loss = eval_loss_sum/len(dev_batch_data)
            eval_acc = eval_acc_sum/len(dev_batch_data)


            print(f"evaluation info: epoch {epoch}, step {global_step}, "
                  f"eval_loss {eval_loss:.6f}, eval_acc {eval_acc:.6f}, "
                  f"eval_time: {eval_time:.2f}s")

            if eval_acc > best_score:
                best_model = copy.deepcopy(model)
                wait = 0
                best_score = eval_acc
            else:
                wait += 1
                if wait >= wait_patient:
                    model = best_model
                    eval_acc = best_score
                    break
            if epoch%50==0:
                torch.save(model, os.path.join(save_dir, 'check_point'+'{}'.format(epoch)+'.bin'))
            result = {
                "epoch": epoch,
                'step': global_step,
                # 'train_loss': train_loss,
                # 'train_acc': train_acc,
                'eval_loss': eval_loss,
                'eval_acc': eval_acc
            }
            output_eval_file = os.path.join(save_dir, "eval_results.txt")
            with open(output_eval_file, "a") as writer:
                for key in sorted(result.keys()):
                    writer.write("%s = %s\n" % (key, str(result[key])))
                writer.write('*' * 80)
                writer.write('\n')
        torch.save(best_model,os.path.join(save_dir,'best_slot_model.bin'))

        total_result = {"loss": 0, "acc": 0, "pre": 0, "rec": 0, "f1": 0}

        for test_data in tqdm(test_batch_data):
            input_ids = test_data['encoder_input'].cuda()
            token_type_ids = test_data['token_type'].cuda()
            attention_mask = test_data['attention_mask'].cuda()
            labels=label_process(test_data['BIO_TAGGING'],label2id,max_length=input_ids.size()[1]).cuda()

            results = self.test(model=model, input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels,token=test_data['input'])
            total_result['loss'] += results['loss']
            total_result['acc'] += results['acc']
            total_result['pre'] += results['pre']
            total_result['rec'] += results['rec']
            total_result['f1'] += results['f1']

        total_result['loss'] = total_result['loss']/len(test_batch_data)
        total_result['acc'] = total_result['acc']/len(test_batch_data)
        total_result['pre'] = total_result['pre']/len(test_batch_data)
        total_result['rec'] = total_result['rec']/len(test_batch_data)
        total_result['f1'] = total_result['f1']/len(test_batch_data)
        total_result['batch_size'] = train_batch_size
        total_result['learning_rate'] = learning_rate
        total_result['best_score'] = best_score


        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        logger.info("Successfully saved slot train model")
        logger.info(total_result)
        logger.info("***************************************************************")

        return total_result, eval_loss_list, epoch_list


    # def eval(self,model,input_ids,token_type_ids,attention_mask,labels):
    def eval(self,
             model: BertSlotFilling,
             input_ids: torch.Tensor,
             token_type_ids: torch.Tensor,
             attention_mask: torch.Tensor,
             labels: torch.Tensor) -> Tuple[float, float]:
        """
        执行验证，在每一个epoch训练完成后调用。

        参数：
            - model：要验证的模型
            - input_ids：验证集编码成的input_ids
            - token_type_ids：验证集编码成的token_type_ids
            - attention_mask：验证集编码成的attention_mask
            - labels：验证集编码成的labels

        返回：
            - eval_loss：在所有验证数据上的平均损失值
            - eval_acc：在所有验证数据上的准确率
        """
        model.eval()
        with torch.no_grad():
            loss, logits = model(input_ids=input_ids, token_type_ids=token_type_ids,
                                 attention_mask=attention_mask, labels=labels)
            num_labels = logits.shape[-1]
            eval_loss = loss.item()
            acc_num = (attention_mask[:, 1:-1].reshape(-1).int() *
                       ((torch.argmax(logits[:, 1:-1, :].reshape(-1, num_labels), dim=-1) ==
                         labels[:, 1:-1].reshape(-1)).int())).sum().item()
            total_num = attention_mask[:, 1:-1].reshape(-1).int().sum().item()
            eval_acc = acc_num/total_num

        return eval_loss, eval_acc


    def test(self,
             model: BertSlotFilling,
             input_ids: torch.Tensor,
             token_type_ids: torch.Tensor,
             attention_mask: torch.Tensor,
             labels: torch.Tensor,token) -> Tuple[float, float]:
        """
        执行验证，在每一个epoch训练完成后调用。

        参数：
            - model：要验证的模型
            - input_ids：验证集编码成的input_ids
            - token_type_ids：验证集编码成的token_type_ids
            - attention_mask：验证集编码成的attention_mask
            - labels：验证集编码成的labels

        返回：
            - eval_loss：在所有验证数据上的平均损失值
            - eval_acc：在所有验证数据上的准确率
        """
        model.eval()
        with torch.no_grad():
            loss, logits = model(input_ids=input_ids, token_type_ids=token_type_ids,
                                 attention_mask=attention_mask, labels=labels)
            # tag = model(input_ids=input_ids, token_type_ids=token_type_ids,
            #                      attention_mask=attention_mask)
            # print(logits.shape)
            num_labels = logits.shape[-1]
            # print("num_labels",num_labels)
            test_loss = loss.item()
            '''
            acc_num = (attention_mask[:, 1:-1].reshape(-1).int() * ((torch.argmax(logits[:, 1:-1, :].reshape(-1, num_labels), dim=-1) ==
                         labels[:, 1:-1].reshape(-1)).int())).sum().item()
            total_num = attention_mask[:, 1:-1].reshape(-1).int().sum().item()
            eval_acc = acc_num/total_num
            '''

            y_predicted = torch.argmax(logits[:, 1:-1, :].reshape(-1, num_labels), dim=-1)
            y_predicted = y_predicted.cpu().numpy()
            y_predicted = y_predicted.tolist()
            # print(token,tag,y_predicted)
            labels = labels[:, 1:-1].reshape(-1)
            labels = labels.cpu().numpy()
            labels = labels.tolist()
            # print("predicted",len(y_predicted))
            # print("labels",len(labels))

            test_acc = accuracy_score(labels, y_predicted)
            test_pre = precision_score(labels, y_predicted, average='weighted')
            test_rec = recall_score(labels, y_predicted, average='weighted')
            test_f1 = f1_score(labels, y_predicted, average='weighted')
            test_confuse = confusion_matrix(labels,y_predicted)

            results = {"loss": test_loss, "acc": test_acc, "pre": test_pre, "rec": test_rec, "f1": test_f1, "confusion_matrix":test_confuse}

        return results


if __name__ == "__main__":
    from config import get_args
    from data_process import prepare_data
    import json
    args = get_args()
    args = vars(args)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    statistic_results = {"ACC": {}, "PRE": {}, "RECALL": {}, "F1 score": {}, "eval_loss": {}, "iteration": {}}
    metric = ["ACC", "PRE", "RECALL", "F1 score"]
    # 槽位标注训练
    slot_train = BertforSlot()
    parameters = {
        # "batch_size":[32,64,80,96,128,160,200,256],
        "batch_size": [128],
        "learning": [0.00005],
    }
    train_loader, val_loader, test_loader, slot_temp, slot_index, slot_type = prepare_data(args, tokenizer)

    description = json.load(open("utils/slot_description.json", 'r'))
    for j in range(len(parameters['learning'])):
        print('length',len(parameters['learning']))
        statistic_results["eval_loss"]["learning rate" + str(parameters["learning"][j])] = {}
        statistic_results["iteration"]["learning rate" + str(parameters["learning"][j])] = {}
        statistic_results["ACC"]["learning rate" + str(parameters["learning"][j])] = []
        statistic_results["PRE"]["learning rate" + str(parameters["learning"][j])] = []
        statistic_results["RECALL"]["learning rate" + str(parameters["learning"][j])] = []
        statistic_results["F1 score"]["learning rate" + str(parameters["learning"][j])] = []
        for i in range(len(parameters['batch_size'])):
            results, eval_loss_list, epoch_list = slot_train.train(
                train_loader, val_loader, test_loader,
                description,
                save_dir='./model/Slot_model/assistant_Slot_Trained_model' + "_"  +'bert_classifier'+'_'+str(parameters["batch_size"][i]) + "_" + str(parameters["learning"][j]), # 模型保存地址
                num_epoches=200,
                train_batch_size=parameters["batch_size"][i],
                learning_rate=parameters["learning"][j],
                tokenizer=tokenizer
            )
            statistic_results["eval_loss"]["learning rate" + str(parameters["learning"][j])]["batch size" + str(parameters["batch_size"][i])] = eval_loss_list
            statistic_results["iteration"]["learning rate" + str(parameters["learning"][j])]["batch size" + str(parameters["batch_size"][i])] = epoch_list
            statistic_results["ACC"]["learning rate" + str(parameters["learning"][j])].append(results["acc"])
            statistic_results["PRE"]["learning rate" + str(parameters["learning"][j])].append(results["pre"])
            statistic_results["RECALL"]["learning rate" + str(parameters["learning"][j])].append(results["rec"])
            statistic_results["F1 score"]["learning rate" + str(parameters["learning"][j])].append(results["f1"])

# train_batch_data,dev_batch_data, test_batch_data, description, save_dir, num_epoches, train_batch_size, learning_rate)
# {'0': 'O', '1': 'B-bool', '2': 'I-bool', '3': 'B-time', '4': 'I-time', '5': 'B-num', '6': 'I-num', '7': 'B-type', '8': 'I-type', '9': 'B-adj', '10': 'I-adj
# ', '11': 'B-area', '12': 'I-area', '13': 'B-day', '14': 'I-day', '15': 'B-name', '16': 'I-name', '17': 'B-location', '18': 'I-location', '19': 'B-food', '2
# 0': 'I-food'}