import torch
import sys
sys.path.append('..')
from transformers.modeling_bert import *


class BERT_EM(BertPreTrainedModel):
    def __init__(self, config):
        super(BERT_EM, self).__init__(config)

        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        
        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.cls.predictions.decoder,
                                   self.bert.embeddings.word_embeddings)

    def forward(self, input_ids, e_pos1, e_pos2, attention_mask = None, token_type_ids = None, position_ids = None, head_mask = None, masked_input_ids = None, masked_lm_labels=None, dropout = torch.nn.Dropout(0)):
        bert_output = self.bert(input_ids,
                            attention_mask = attention_mask,
                            token_type_ids = token_type_ids,
                            position_ids = position_ids,
                            head_mask = head_mask)

        if masked_input_ids is not None:
            masked_bert_output = self.bert(masked_input_ids,
                            attention_mask = attention_mask,
                            token_type_ids = token_type_ids,
                            position_ids = position_ids,
                            head_mask = head_mask)
            masked_sequence_output = masked_bert_output[0]

            # calculate mlm_loss
            prediction_scores = self.cls(masked_sequence_output)
            masked_bert_output = (prediction_scores,) + masked_bert_output[2:]
            
            if masked_lm_labels is not None:
                loss_fct = CrossEntropyLoss(ignore_index=-1)
                masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
        

        # get relation representation
        sequence_output = bert_output[0] # batch_size * sequence_length * hidden_size
        factor = torch.tensor(range(0, e_pos1.shape[0])).cuda()
        unit = input_ids.shape[1]
        offset = factor * unit
        e_pos1 = e_pos1 + offset
        e_pos2 = e_pos2 + offset

        start_embedding_e1 = torch.index_select(sequence_output.view(-1, sequence_output.shape[2]), 0, e_pos1) # batch_size * hidden_size
        start_embedding_e2 = torch.index_select(sequence_output.view(-1, sequence_output.shape[2]), 0, e_pos2)
        relation_embedding = torch.cat((start_embedding_e1, start_embedding_e2), 1)
        if torch.cuda.is_available():
            start_embedding_e1 = start_embedding_e1.cuda()
            start_embedding_e2 = start_embedding_e2.cuda()
            relation_embedding = relation_embedding.cuda()

        return masked_lm_loss, relation_embedding
