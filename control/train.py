import torch
import sys
sys.path.append('..')
import torch.optim as optim
from models.bert_em import BERT_EM
from models.proto_sim import ProtoSimModel 
from torch.utils.data import DataLoader
from transformers.tokenization_bert import BertTokenizer
from dataset.dataset import Data
from transformers import AdamW, WarmupLinearSchedule
import os
import json


os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device_ids = [0]

def flat(batch):
    # batch, sample_number, data -> batch * sample_number, data
    size = batch.size()
    if len(size) == 3:
        return batch.view(-1, size[-1])
    elif len(size) == 2:
        return batch.view(size[0] * size[1])

# This func is copied from transformers
def mask_tokens(inputs, tokenizer):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, 0.15)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -1  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

def train(train_data, tokenizer, bert_encoder, config):
    print('Start training...')
    data_loader = DataLoader(train_data, batch_size = 1, shuffle = True)
    grad_iter = config.get('grad_iter', 16)
    proto_sim_model = ProtoSimModel(config['relations'], config['embedding_size'])
    if torch.cuda.is_available():
        bert_encoder = bert_encoder.cuda()
        proto_sim_model = proto_sim_model.cuda()

    # setup optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{'params': [p for n, p in bert_encoder.named_parameters() if not any(nd in n for nd in no_decay)] + [p for n, p in proto_sim_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0 },{'params': [p for n, p in bert_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, eps=1e-8)
    t_total = int(config['iterations'] * data_loader.__len__() / grad_iter) # total batches
    warmup_steps = int(int(t_total * 0.06) / grad_iter) 
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)

    cross_entropy = torch.nn.CrossEntropyLoss()
    bert_encoder.train()
    bert_encoder.zero_grad()
    proto_sim_model.zero_grad()

    reversed_index = list(range(8))
    reversed_index.reverse()

    for i in range(config['iterations']):
        print('Iteration: # %d / %d' % (i, config['iterations']))
        for i_batch, batch_data in enumerate(data_loader):
            # p_ = positive , n_ = negative
            p_input_ids = flat(batch_data['p_input_id'])
            p_masked_input_ids, p_masked_mlm_labels = mask_tokens(p_input_ids, tokenizer)
            p_mask = flat(batch_data['p_mask'])
            p_e_pos1 = flat(batch_data['p_e_pos1'])
            p_e_pos2 = flat(batch_data['p_e_pos2'])
            p_labels = flat(batch_data['p_label'])
            p_mlm_loss, p_relation_embedding = bert_encoder(p_input_ids.cuda(), p_e_pos1.cuda(), p_e_pos2.cuda(), attention_mask = p_mask.cuda(), masked_input_ids = p_masked_input_ids.cuda(), masked_lm_labels = p_masked_mlm_labels.cuda())
            p_similarity, p_predict_relation = proto_sim_model(p_relation_embedding, p_labels.cuda())

            n_input_ids = flat(batch_data['n_input_id'])
            n_masked_input_ids, n_masked_mlm_labels = mask_tokens(n_input_ids, tokenizer)
            n_mask = flat(batch_data['n_mask'])
            n_e_pos1 = flat(batch_data['n_e_pos1'])
            n_e_pos2 = flat(batch_data['n_e_pos2'])
            n_mlm_loss, n_relation_embedding = bert_encoder(n_input_ids.cuda(), n_e_pos1.cuda(), n_e_pos2.cuda(), attention_mask = n_mask.cuda(), masked_input_ids = n_masked_input_ids.cuda(), masked_lm_labels = n_masked_mlm_labels.cuda())
            n_similarity, n_predict_relation = proto_sim_model(n_relation_embedding, p_labels.cuda())
            cluster_loss = -(torch.mean(torch.log(p_similarity + 1e-5)) + torch.mean(torch.log(1-n_similarity + 1e-5)))
            cls_loss = torch.mean(cross_entropy(p_predict_relation, p_labels.cuda()) + cross_entropy(n_predict_relation, p_labels.cuda())) * 0.5
            mlm_loss = p_mlm_loss.mean() + n_mlm_loss.mean()

            # Contrastive loss
            # (batch_size, sample_size, embedding_size)
            repeat_time, embedding_size = n_relation_embedding.shape
            repeat_n_rel_emb = n_relation_embedding.repeat(repeat_time, 1).view(repeat_time, repeat_time, embedding_size)
            repeat_p_rel_emb = p_relation_embedding.repeat(1, repeat_time).view(repeat_time, repeat_time, embedding_size)
            negative_part = torch.sum(repeat_p_rel_emb * repeat_n_rel_emb, -1) / 100 # (batch_size, sample_size)
            reversed_p_relation_embedding = torch.index_select(p_relation_embedding.view(1, -1, config['embedding_size']), 1, torch.tensor(reversed_index).cuda()).view(-1, config['embedding_size'])
            positive_part = torch.sum(p_relation_embedding * reversed_p_relation_embedding, -1) / 100
            max_val = torch.max(torch.max(positive_part), torch.max(negative_part))
            cp_loss = -torch.log(torch.exp(positive_part - max_val) / (torch.exp(positive_part - max_val) + torch.sum(torch.exp(negative_part - max_val), -1)) + 1e-5)
            cp_loss = torch.mean(cp_loss)

            loss = 0.5 * cluster_loss + 0.5 * cls_loss + cp_loss + mlm_loss * 0.5
            loss = loss / float(grad_iter)
            if i_batch % 500 == 0:
                print('cluster_loss: ', cluster_loss)
                print('cls_loss: ', cls_loss)
                print('cp_loss: ', cp_loss)
                print('mlm_loss: ', mlm_loss)
                print('loss: ', loss)
            loss.backward()
            if i_batch % grad_iter == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        if (i + 1) % config['save_interval'] == 0:
            model_dir = './ckpts/ckpt_%d' % i
            if not os.path.exists(model_dir):
                os.mkdir(model_dir)
            bert_encoder.save_pretrained(model_dir)


if __name__ == '__main__':
    train_config = json.load(open(sys.argv[1]))
    train_data = Data(train_config['train_data'])
    tokenizer = BertTokenizer(train_config['vocab'], do_lower_case = False)
    bert_encoder = BERT_EM.from_pretrained(train_config['base_model'])
    print('Finished loading pre-trained model...')
    train(train_data, tokenizer, bert_encoder, train_config)
