# -*- coding: utf-8 -*-
# @description:
# @author: zchen
# @time: 2020/11/30 18:23
# @file: models.py.py

import torch.nn as nn
from torchcrf import CRF
from transformers import BertPreTrainedModel, BertModel


class BERT_BiLSTM_CRF(BertPreTrainedModel):

    def __init__(self, config, need_birnn=False, rnn_dim=128):
        super(BERT_BiLSTM_CRF, self).__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        out_dim = config.hidden_size
        
        if need_birnn:
            self.need_birnn = need_birnn
            self.birnn = nn.LSTM(input_size=config.hidden_size, hidden_size=rnn_dim, num_layers=1, bidirectional=True,
                                 batch_first=True)
            out_dim = rnn_dim * 2

        self.hidden2tag = nn.Linear(in_features=out_dim, out_features=config.num_labels)

        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

    def forward(self, input_ids, tags, token_type_ids=None, attention_mask=None):
 
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        if self.need_birnn:
            sequence_output, _ = self.birnn(sequence_output)
        sequence_output = self.dropout(sequence_output)
        emissions = self.hidden2tag(sequence_output)
        loss = -1 * self.crf(emissions, tags, mask=attention_mask.byte())
        return loss

    def predict(self, input_ids, token_type_ids=None, attention_mask=None):

        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        if self.need_birnn:
            sequence_output, _ = self.birnn(sequence_output)
        sequence_output = self.dropout(sequence_output)
        emissions = self.hidden2tag(sequence_output)
        return self.crf.decode(emissions, attention_mask.byte())


"""
BertConfig {
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "classifier_dropout": null,
  "directionality": "bidi",
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "id2label": {
    "0": "LABEL_0",
    "1": "LABEL_1",
    "2": "LABEL_2"
  },
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "label2id": {
    "LABEL_0": 0,
    "LABEL_1": 1,
    "LABEL_2": 2
  },
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "pooler_fc_size": 768,
  "pooler_num_attention_heads": 12,
  "pooler_num_fc_layers": 3,
  "pooler_size_per_head": 128,
  "pooler_type": "first_token_transform",
  "position_embedding_type": "absolute",
  "transformers_version": "4.30.2",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 21128
}
"""


"""
BERT_BiLSTM_CRF(
  (bert): BertModel(
    (embeddings): BertEmbeddings(
      (word_embeddings): Embedding(21128, 768, padding_idx=0)
      (position_embeddings): Embedding(512, 768)
      (token_type_embeddings): Embedding(2, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): BertEncoder(
      (layer): ModuleList(
        (0-11): 12 x BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
         )
      )
    )
    (pooler): BertPooler(
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (activation): Tanh()
    )
  )
  (dropout): Dropout(p=0.1, inplace=False)
  (birnn): LSTM(768, 128, batch_first=True, bidirectional=True)
  (hidden2tag): Linear(in_features=256, out_features=3, bias=True)
  (crf): CRF(num_tags=3)
)


"""