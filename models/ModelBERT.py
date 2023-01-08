import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel
from sympy import *


class UtteranceEncoding(BertPreTrainedModel):
    def __init__(self, config):
        super(UtteranceEncoding, self).__init__(config)

        self.config = config
        self.bert = BertModel(config)

        self.init_weights()

    def forward(self, input_ids, attention_mask, token_type_ids, output_attentions=False, output_hidden_states=False):
        return self.bert(input_ids=input_ids,
                         attention_mask=attention_mask,
                         token_type_ids=token_type_ids,
                         output_attentions=output_attentions,
                         output_hidden_states=output_hidden_states)


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

        self.scores = None

    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)

        self.scores = scores
        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output

    def get_scores(self):
        return self.scores


class PowerLawAttention(nn.Module):
    def __init__(self, heads, d_model, device, dropout=0.1, num_turns=30):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

        self.scores = None
        self.device = device

        self.b_encoder = nn.Embedding(num_turns**2, heads, padding_idx=0)
        self.grama_list = self.grama(num_turns)

    def grama(self, num_turns):
        grama_list = []
        for idx in range(num_turns):
            if idx == 0:
                grama_history = 1
            else:
                t = symbols('t')
                grama_history = integrate(1 / (t + 1), (t, idx - 1, idx))
            grama_list.append(grama_history)
        grama_list.sort()
        return grama_list

    def list_soft_max(self, input_list):
        result = []
        denominator = 0
        for data in input_list:
            denominator += math.exp(data)
        for data in input_list:
            result.append(math.exp(data) / denominator)
        return result

    def power_law_distribution(self, input_token_turn_list, history_type_turn_id_list, slot_type):
        batch_size = input_token_turn_list.size(0)
        max_len = input_token_turn_list.size(1)
        slot_num = len(slot_type)

        batch_grama_list = np.zeros((batch_size, slot_num, max_len))

        for i in range(batch_size):
            this_turn_idx = np.max(input_token_turn_list[i])
            grama_history_result = self.list_soft_max(self.grama_list[-(this_turn_idx+1):])
            #print("^^^^")
            #print(input_token_turn_list[i])
            for j in range(slot_num):
                batch_grama_list[i, j, :] = [grama_history_result[idx] if idx != -1 else 0 for idx in input_token_turn_list[i]]

        for i, dict in enumerate(history_type_turn_id_list): #batch
            for j, type in enumerate(slot_type): #slot
                #print("^^^^")
                #print(dict)
                for key, turn_id_list in enumerate(dict):
                    if type == key:
                        grama_type_history = self.list_soft_max(self.grama_list[-len(turn_id_list):])

                        for d, id in turn_id_list:
                            for k, turn_idx in enumerate(input_token_turn_list):
                                if turn_idx == id:
                                    batch_grama_list[i][j][k] += grama_type_history[d]

        batch_grama_tensor = torch.LongTensor(batch_grama_list).to(self.device)
        #print("^^^^")
        #print(batch_grama_tensor.size())
        #batch_grama_tensor = batch_grama_tensor.unsqueeze(1).repeat(1, 4, 1, 1)
        print("-----b------")
        #print(b.size())
        print(batch_grama_tensor.size())
        return batch_grama_tensor

    def attention(self, q, k, v, b, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        #print("-----score------")
        #print(scores.size())
        b = b.permute(0, 3, 1, 2).contiguous()
        scores += b

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)

        self.scores = scores
        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v, input_token_turn_list, history_type_turn_id_list, slot_type, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        b = self.b_encoder(self.power_law_distribution(input_token_turn_list, history_type_turn_id_list, slot_type))
        print("-----b encode------")
        print(b.size())
        scores = self.attention(q, k, v, b, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output

    def get_scores(self):
        return self.scores

class MultiHeadAttentionTanh(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

        self.scores = None

    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        scores = torch.tanh(scores)
        #         scores = torch.sigmoid(scores)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, 0.)
        #         scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)

        self.scores = scores
        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output

    def get_scores(self):
        return self.scores


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([deepcopy(module) for _ in range(N)])


class SlotSelfAttention(nn.Module):
    "A stack of N layers"

    def __init__(self, layer, N):
        super(SlotSelfAttention, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, mask=None):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        x = self.norm(x)
        return x + self.dropout(sublayer(x))


class SlotAttentionLayer(nn.Module):
    "SlotAttentionLayer is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(SlotAttentionLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.ReLU()  # use gelu or relu

    def forward(self, x):
        return self.w_2(self.dropout(self.gelu(self.w_1(x))))


class UtteranceAttention(nn.Module):
    def __init__(self, attn_head, model_output_dim, device, dropout=0., attn_type="softmax"):
        super(UtteranceAttention, self).__init__()
        self.attn_head = attn_head
        self.model_output_dim = model_output_dim
        self.dropout = dropout
        self.attn_type = attn_type
        self.device = device
        if self.attn_type == "tanh":
            self.attn_fun = MultiHeadAttentionTanh(self.attn_head, self.model_output_dim, dropout=0.)
        elif self.attn_type == "power-law":
            self.attn_fun = PowerLawAttention(self.attn_head, self.model_output_dim, self.device, dropout=0.)
        else:
            self.attn_fun = MultiHeadAttention(self.attn_head, self.model_output_dim, dropout=0.)

    def forward(self, query, value, attention_mask=None, input_token_turn_list=None, history_type_turn_id_list=None, slot_type=None):
        num_query = query.size(0)
        batch_size = value.size(0)
        seq_length = value.size(1)

        expanded_query = query.unsqueeze(0).expand(batch_size, *query.shape)
        if attention_mask is not None:
            expanded_attention_mask = attention_mask.view(-1, seq_length, 1).expand(value.size()).float()
            new_value = torch.mul(value, expanded_attention_mask)
            attn_mask = attention_mask.unsqueeze(1).expand(batch_size, num_query, seq_length)
        else:
            new_value = value
            attn_mask = None
        if self.attn_type == "power-law":
            attended_embedding = self.attn_fun(expanded_query, new_value, new_value,
                                               input_token_turn_list, history_type_turn_id_list, slot_type, mask=attn_mask)
        else:
            attended_embedding = self.attn_fun(expanded_query, new_value, new_value, mask=attn_mask)

        return attended_embedding


class FusionGate(nn.Module):
    def __init__(self,model_output_dim):
        # 初始化函数
        super(FusionGate, self).__init__()
        self.activate_function = nn.Sigmoid()
        self.W_mlp = nn.Linear(model_output_dim * 2, model_output_dim)

    def forward(self, sequence_output, state_output):
        sigma =self.activate_function(self.W_mlp(torch.cat((sequence_output, state_output), -1)))
        fusion_result = (1 - sigma) * sequence_output + sigma * state_output
        return fusion_result

class Decoder(nn.Module):
    def __init__(self, args, model_output_dim, num_labels, slot_value_pos, device):
        super(Decoder, self).__init__()
        self.model_output_dim = model_output_dim
        self.num_slots = len(num_labels)
        self.num_total_labels = sum(num_labels)
        self.num_labels = num_labels
        self.slot_value_pos = slot_value_pos
        self.attn_head = args.attn_head
        self.device = device
        self.args = args
        self.dropout_prob = self.args.dropout_prob
        self.dropout = nn.Dropout(self.dropout_prob)
        self.attn_type = self.args.attn_type

        ### slot utterance attention
        self.slot_state_attn = UtteranceAttention(self.attn_head, self.model_output_dim, self.device, dropout=0.,
                                                  attn_type=self.attn_type)
        self.slot_history_attn = UtteranceAttention(self.attn_head, self.model_output_dim, self.device, dropout=0.,
                                                  attn_type="power-law")
        ### MLP
        self.SlotMLP = nn.Sequential(nn.Linear(self.model_output_dim * 2, self.model_output_dim),
                                     nn.ReLU(),
                                     nn.Dropout(p=self.dropout_prob),
                                     nn.Linear(self.model_output_dim, self.model_output_dim))
        self.gate = FusionGate(self.model_output_dim)

        ### basic modues, attention dropout is 0.1 by default
        attn = MultiHeadAttention(self.attn_head, self.model_output_dim)
        ffn = PositionwiseFeedForward(self.model_output_dim, self.model_output_dim, self.dropout_prob)

        ### attention layer, multiple self attention layers
        self.slot_self_attn = SlotSelfAttention(SlotAttentionLayer(self.model_output_dim, deepcopy(attn),
                                                                   deepcopy(ffn), self.dropout_prob),
                                                self.args.num_self_attention_layer)

        ### prediction
        self.pred = nn.Sequential(nn.Dropout(p=self.dropout_prob),
                                  nn.Linear(self.model_output_dim, self.model_output_dim),
                                  nn.LayerNorm(self.model_output_dim))

        ### measure
        self.distance_metric = args.distance_metric
        if self.distance_metric == "cosine":
            self.metric = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)
        elif self.distance_metric == "euclidean":
            self.metric = torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)

        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)
        self.nll = CrossEntropyLoss(ignore_index=-1)

    def slot_value_matching(self, value_lookup, hidden, target_slots, labels):
        loss = 0.
        loss_slot = []
        pred_slot = []

        batch_size = hidden.size(0)
        value_emb = value_lookup.weight[0:self.num_total_labels, :]

        for s, slot_id in enumerate(target_slots):  # note: target_slots are successive
            hidden_label = value_emb[self.slot_value_pos[slot_id][0]:self.slot_value_pos[slot_id][1], :]
            num_slot_labels = hidden_label.size(0)  # number of value choices for each slot

            _hidden_label = hidden_label.unsqueeze(0).repeat(batch_size, 1, 1).reshape(batch_size * num_slot_labels, -1)
            _hidden = hidden[:, s, :].unsqueeze(1).repeat(1, num_slot_labels, 1).reshape(batch_size * num_slot_labels,
                                                                                         -1)
            _dist = self.metric(_hidden_label, _hidden).view(batch_size, num_slot_labels)

            if self.distance_metric == "euclidean":
                _dist = -_dist

            _, pred = torch.max(_dist, -1)
            pred_slot.append(pred.view(batch_size, 1))

            _loss = self.nll(_dist, labels[:, s])

            loss += _loss
            loss_slot.append(_loss.item())

        pred_slot = torch.cat(pred_slot, 1)  # [batch_size, num_slots]

        return loss, loss_slot, pred_slot

    def forward(self, sequence_output, state_output, attention_mask, input_mask_state, labels, slot_lookup, value_lookup,
                input_token_turn_list, history_type_turn_id_list, slot_type, eval_type="train"):

        batch_size = sequence_output.size(0)
        target_slots = list(range(0, self.num_slots))

        # slot utterance attention
        slot_embedding = slot_lookup.weight[target_slots, :]  # select target slots' embeddings
        slot_utter_emb = self.slot_history_attn(slot_embedding, sequence_output, attention_mask,
                                                input_token_turn_list, history_type_turn_id_list, slot_type)
        #print(slot_utter_emb.size())

        slot_state_emb = self.slot_state_attn(slot_embedding, state_output, input_mask_state)
        #print(slot_state_emb.size())
        # concatenate with slot_embedding
        slot_utter_embedding = torch.cat((slot_embedding.unsqueeze(0).repeat(batch_size, 1, 1), slot_utter_emb), 2)
        slot_state_embedding = torch.cat((slot_embedding.unsqueeze(0).repeat(batch_size, 1, 1), slot_state_emb), 2)

        # MLP
        slot_utter_embedding2 = self.SlotMLP(slot_utter_embedding)
        slot_state_embedding2 = self.SlotMLP(slot_state_embedding)

        hidden_state = self.gate(slot_utter_embedding2, slot_state_embedding2)

        # slot self attention
        hidden_slot = self.slot_self_attn(hidden_state)

        # prediction
        hidden = self.pred(hidden_slot)

        # slot value matching
        loss, loss_slot, pred_slot = self.slot_value_matching(value_lookup, hidden, target_slots, labels)

        return loss, loss_slot, pred_slot


class BeliefTracker(nn.Module):
    def __init__(self, args, slot_lookup, value_lookup, num_labels, slot_value_pos, device):
        super(BeliefTracker, self).__init__()

        self.num_slots = len(num_labels)
        self.num_labels = num_labels
        self.slot_value_pos = slot_value_pos
        self.args = args
        self.device = device
        self.slot_lookup = slot_lookup
        self.value_lookup = value_lookup

        self.encoder = UtteranceEncoding.from_pretrained(self.args.pretrained_model)
        self.model_output_dim = self.encoder.config.hidden_size
        self.decoder = Decoder(args, self.model_output_dim, self.num_labels, self.slot_value_pos, device)

    def forward(self, input_ids, attention_mask, token_type_ids, input_ids_state, input_mask_state, segment_ids_state, labels,
                input_token_turn_list, history_type_turn_id_list, slot_type, eval_type="train"):
        batch_size = input_ids.size(0)
        num_slots = self.num_slots

        # encoder, a pretrained model, output is a tuple
        sequence_output = self.encoder(input_ids, attention_mask, token_type_ids)[0]

        state_output = self.encoder(input_ids_state, input_mask_state, segment_ids_state)[0]
        state_output = state_output.detach()

        # decoder
        loss, loss_slot, pred_slot = self.decoder(sequence_output, state_output, attention_mask, input_mask_state,
                                                  labels, self.slot_lookup, self.value_lookup,
                                                  input_token_turn_list, history_type_turn_id_list, slot_type, eval_type)

        # calculate accuracy
        accuracy = pred_slot == labels
        acc_slot = torch.true_divide(torch.sum(accuracy, 0).float(), batch_size).cpu().detach().numpy()  # slot accuracy
        acc = torch.sum(
            torch.floor_divide(torch.sum(accuracy, 1), num_slots)).float().item() / batch_size  # joint accuracy

        return loss, loss_slot, acc, acc_slot, pred_slot