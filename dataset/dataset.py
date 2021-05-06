#coding:utf-8

import sys
import torch
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

MAX_LENGTH = 128

"""
Data format:<instance id>\001<relation id>\001<start position of the first entity>\001<start position of the second entity>\001<sentence (sequence of word ids)>
"""

class Data(Dataset):

    def __init__(self, data_path):
        print('Loading data...')
        self.relation_name_id, self.relation_inst_ids, self.inst_id_detail = self.load_data(data_path)
        print('Finish loading data.')


    def load_data(self, data_path):
        relation_name_id = {}
        relation_inst_ids = {}
        inst_id_detail = {}
        for line in open(data_path):
            inst_id, r, e_pos1, e_pos2, word_ids = line.strip().split('\001')
            e_pos1 = int(e_pos1)
            e_pos2 = int(e_pos2)
            if r not in relation_name_id:
                relation_name_id[r] = len(relation_name_id.keys())
            word_ids = [int(x) for x in word_ids.split('\002')]
            detail = (word_ids, e_pos1, e_pos2, relation_name_id[r])
            inst_id_detail[inst_id] = detail
            inst_list = relation_inst_ids.get(relation_name_id[r], [])
            inst_list.append(inst_id)
            relation_inst_ids[relation_name_id[r]] = inst_list
        return relation_name_id, relation_inst_ids, inst_id_detail

        
    def sample(self, item_list, count):
        if len(item_list) < count:
            idx = item_list[0]
            idxs = [idx] * count
        else:
            idxs = random.sample(item_list, count)
        return idxs


    def collect(self, dict_list, idxs):
        input_ids, e_pos1, e_pos2, label = [], [], [], []
        for idx in idxs:
            input_ids.append(dict_list[idx][0])
            e_pos1.append(dict_list[idx][1])
            e_pos2.append(dict_list[idx][2])
            label.append(dict_list[idx][3])
        return input_ids, e_pos1, e_pos2, label
    

    def get_mask(self, id_list):
        mask = []
        for l in id_list:
            mask.append([1] * len(l))
        return mask

    def padding(self, item_list, max_length, default):
        new_out = []
        for l in item_list:
            l = l[:max_length]
            l.extend([default] * (max_length - len(l)))
            new_out.append(l)
        return new_out

    def __len__(self):
        return len(self.relation_inst_ids.keys())


    def __getitem__(self, idx):
        relations = random.sample(self.relation_inst_ids.keys(), 5)
        positive_relation = relations[0]
        idxs = self.sample(self.relation_inst_ids[positive_relation], 8)
        p_input_ids, p_e_pos1, p_e_pos2, p_label = self.collect(self.inst_id_detail, idxs)
        p_mask = self.get_mask(p_input_ids)
        p_input_ids = self.padding(p_input_ids, MAX_LENGTH, 0)
        p_mask = self.padding(p_mask, MAX_LENGTH, 0)

        idxs = []
        for r in relations[1:]:
            idxs.extend(self.sample(self.relation_inst_ids[r], 2))

        n_input_ids, n_e_pos1, n_e_pos2, n_label = self.collect(self.inst_id_detail, idxs)
        n_mask = self.get_mask(n_input_ids)
        n_input_ids = self.padding(n_input_ids, MAX_LENGTH, 0)
        n_mask = self.padding(n_mask, MAX_LENGTH, 0)

        p_input_ids, p_mask, p_e_pos1, p_e_pos2, p_label, n_input_ids, n_mask, n_e_pos1, n_e_pos2, n_label = map(lambda x : torch.tensor(x), [p_input_ids, p_mask, p_e_pos1, p_e_pos2, p_label, n_input_ids, n_mask, n_e_pos1, n_e_pos2, n_label])

        sample = {'p_input_id': p_input_ids, 'p_mask': p_mask, 'p_e_pos1': p_e_pos1, 'p_e_pos2': p_e_pos2, 'p_label':p_label, 'n_input_id': n_input_ids, 'n_mask': n_mask, 'n_e_pos1': n_e_pos1, 'n_e_pos2': n_e_pos2, 'n_label':n_label}
        return sample


if __name__ == '__main__':
    data = Data(sys.argv[1])
    data_loader = DataLoader(data, batch_size = 1, shuffle = True)
    for i_batch, batch_data in enumerate(data_loader):
        print(batch_data)
