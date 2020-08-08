import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertForSequenceClassification, BertModel


class BertForEntityClassification(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        
        self.num_labels = config.num_labels

        if config.tags:
            self.spec_tag1, self.spec_tag2, self.spec_tag3, self.spec_tag4 = config.tags
        else:
            self.spec_tag1, self.spec_tag2, self.spec_tag3, self.spec_tag4 = None, None, None, None

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier1 = nn.Linear(config.hidden_size, config.num_labels)
        self.classifier2 = nn.Linear(config.hidden_size*2, config.num_labels)

        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                **kwargs):
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        if self.spec_tag1:
            seq_output = outputs[0]
            seq_output = self.dropout(seq_output)
            spec_idx = (input_ids == self.spec_tag1).nonzero()
            temp = []
            for idx in spec_idx:
                temp.append(seq_output[idx[0], idx[1], :])
            seq_tag = torch.stack(temp)
            new_pooled_output = torch.cat((pooled_output, seq_tag), 1)
            # print(new_pooled_output.shape)
            logits = self.classifier2(new_pooled_output)
        else:
            logits = self.classifier1(pooled_output)

        outputs = (logits,) + outputs[2:]

        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
            outputs = (loss,) + outputs
            
        return outputs


class BertForRelationIdentification(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        
        if config.tags:
            self.spec_tag1, self.spec_tag2, self.spec_tag3, self.spec_tag4 = config.tags
        else:
            self.spec_tag1, self.spec_tag2, self.spec_tag3, self.spec_tag4 = None, None, None, None
        
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier1 = nn.Linear(config.hidden_size, config.num_labels)
        self.classifier2 = nn.Linear(config.hidden_size*3, config.num_labels)

        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                **kwargs):
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        if self.spec_tag1:
            seq_output = outputs[0]
            seq_output = self.dropout(seq_output)
            
            spec_idx1 = (input_ids == self.spec_tag1).nonzero()
            temp1 = []
            for idx in spec_idx1:
                temp1.append(seq_output[idx[0], idx[1], :])
            seq_tag1 = torch.stack(temp1)
            
            spec_idx2 = (input_ids == self.spec_tag3).nonzero()
            temp2 = []
            for idx in spec_idx2:
                temp2.append(seq_output[idx[0], idx[1], :])
            seq_tag2 = torch.stack(temp2)
     
            new_pooled_output = torch.cat((pooled_output, seq_tag1, seq_tag2), 1)

            logits = self.classifier2(new_pooled_output)
        else:
            logits = self.classifier1(pooled_output)

        outputs = (logits,) + outputs[2:]

        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
            outputs = (loss,) + outputs
            
        return outputs