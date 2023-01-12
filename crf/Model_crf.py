import os
import torch
import random
import time
from transformers import BertModel, BertForSequenceClassification, BertTokenizer, BertConfig, BertForTokenClassification
from allennlp.modules.conditional_random_field import ConditionalRandomField, allowed_transitions

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class BertSlotFilling(torch.nn.Module):
    def __init__(self, transformers_model, id2label):
        super(BertSlotFilling, self).__init__()
        self.transformers_model = transformers_model

        crf_constraints = allowed_transitions(
            constraint_type="BIO",
            labels=id2label
        )
        self.crf = ConditionalRandomField(
            num_tags=len(id2label),
            constraints=crf_constraints,
            include_start_end_transitions=True
        )
        unfreeze_layers = ['classifier']
        for name, param in self.transformers_model.named_parameters():
            # print('name',name)
            param.requires_grad = False
            for ele in unfreeze_layers:
               if ele in name:
                 param.requires_grad = True
                 print('name',name)
                 break

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        if labels is not None:
            outputs = self.transformers_model(input_ids=input_ids, token_type_ids=token_type_ids,
                                              attention_mask=attention_mask, labels=labels)

            # print('outputs',len(outputs))
            _, logits = outputs[:2]
            loss = -1.0 * self.crf(logits[:, 1:-1, :], labels[:, 1:-1], attention_mask[:, 1:-1])
            return loss, logits
        else:
            outputs = self.transformers_model(input_ids=input_ids, token_type_ids=token_type_ids,
                                              attention_mask=attention_mask)
            logits = outputs[0]
            best_paths = self.crf.viterbi_tags(logits[:, 1:-1, :], attention_mask[:, 1:-1])
            predicted_tags = [x for x, y in best_paths]
            return predicted_tags

    def save_pretrained(self, path):
        self.transformers_model.save_pretrained(path)
        torch.save(self.crf.state_dict(), os.path.join(path, "crf.bin"))

    @classmethod
    def from_pretrained(cls, path):
        transformers_model = BertForTokenClassification.from_pretrained(path)
        id2label = transformers_model.config.id2label
        assert type(list(id2label.keys())[0]) == int
        model = cls(transformers_model, id2label)
        model.crf.load_state_dict(torch.load(os.path.join(path, "crf.bin")))
        return model
