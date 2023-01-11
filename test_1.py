from transformers import BertTokenizer
import torch
from models.ModelBERT import UtteranceEncoding

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
label = "none"
x = tokenizer(label)["input_ids"]
label_token_ids = torch.LongTensor(x).reshape(1,-1)
label_type_ids = torch.LongTensor([0] * label_token_ids.size(0)).reshape(1,-1)
label_mask = torch.LongTensor([1 if i>0 else 0 for i in x]).reshape(1,-1)

sv_encoder = UtteranceEncoding.from_pretrained('bert-base-uncased')
sv_encoder.eval()
value_lookup_1 = sv_encoder(label_token_ids, label_mask, label_type_ids)[1]
print(value_lookup_1)

#print(x)
#y = x.append(0)
y = [101, 3904, 102, 0, 0]
#print(y)
label_token_ids = torch.LongTensor(y).reshape(1,-1)
label_type_ids = torch.LongTensor([0] * label_token_ids.size(0)).reshape(1,-1)
label_mask = torch.LongTensor([1 if i>0 else 0 for i in y]).reshape(1,-1)

#sv_encoder = UtteranceEncoding.from_pretrained('bert-base-uncased')
value_lookup_1 = sv_encoder(label_token_ids, label_mask, label_type_ids)[1]
print(value_lookup_1)
