import torch
import pandas as pd
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel
from transformers import logging
logging.set_verbosity_error()


def bert_enc(text):
    # 初始化 BERT tokenizer 和模型
    # medicalai/ClinicalBERT
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # model = BertModel.from_pretrained('bert-base-uncased')

    tokenizer = BertTokenizer.from_pretrained('medicalai/ClinicalBERT')
    model = BertModel.from_pretrained('medicalai/ClinicalBERT')

    # tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    # model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    # tokenization
    input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])

    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]

    sentence_embedding = last_hidden_states[0][0]

    return torch.nn.functional.normalize(sentence_embedding, p=2, dim=0)

# text = "The body mass index is 37, there is no hypertension, no diabetes mellitus, no heart disease, no hyperlipoidemia, and the age is 50."
# short_text = "The body mass index is 37, and the age is 50."
# print(bert_enc(text).shape)
# print(bert_enc(short_text).shape)


