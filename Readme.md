ProtoRE

Introduction

This repo contains the code of the pretraining method proposed in  Prototypical Representation Learning for Relation Extraction Ning Ding*, Xiaobin Wang*, Yao Fu, Guangwei Xu, Rui Wang, Penguin Xie, Ying Shen, Fei Huang, Hai-Tao Zheng, Rui Zhang. ICLR 2021.

Prerequisites

Environment

- python : 3.7.5
- pytorch : 1.3.1
- tranformers : 2.1.1

Key Implementation

The relation representation produced by BERT is implemented in models/bert_em.py;

Equation (3) in the paper is implemented in control/train.py line 103-108;

Equation (4) (5) (6) in the paper is implemented in control/train.py line 78-96;

Prototypes are implemented in models/proto_sim.py.

Data

Distantly labeled training data is required for runing. Training data should have five columns, i.e. instance id, relation id, start position of the first entity, start position of the second entity and a sentence. The sentence is converted to a sequence of word ids by BertTokenizer. Special tokens like [CLS] and [SEP] are added (So a sequence starts with 101 and ends with 102). Reserved tokens are used as entity markers, [unused1] -> <e1>, [unused2] -> </e1>, [unused2] -> <e2>, [unused4] -> </e2>. There is a sample.txt in directory data for demo.

Run

First, modify the train_config.json according your data and setting.

- train_data : path of train data
- vocab : path of the vocabulary file for BertTokenizer
- base_model : path of basic pretrained model, e.g., bert-base-cased
- relations : number of relations in training data,
- embedding_size : size of relation representation,
- iterations  : training iterations,
- save_interval : the interval of saving checkpoints

Second, create a directory for saving models.

    mkdir ckpts

Then, run the code:

    python train.py train_config.json

It will take a long time (serval days in one tesla P100) for training. 

How to use the pretrained encoder

First, load the pretrained model

    from models.bert_em import BERT_EM
    bert_encoder = BERT_EM.from_pretrained("path/to/saved_model_dir")

Second, feed inputs and get relation embeddings.

    """
    p_input_ids: Indices of input sequence tokens in the vocabulary.
    attention_mask: Mask to avoid performing attention on padding token indices.
    e1_pos: positiion of entity start marker of the first entity (zero based, CLS included).
    e2_pos: positiion of entity start marker of the second entity.
    p_input_ids, attention_mask is same as the input of BertModel in transformers.
    """
    _, rel_emb = bert_encoder(p_input_ids, e1_pos, e2_pos, attention_mask = mask)



Citation

Please cite our paper is you use prototypical method in your work:

    @article{ding2021prototypical,
      title={Prototypical Representation Learning for Relation Extraction},
      author={Ding, Ning and Wang, Xiaobin and Fu, Yao and Xu, Guangwei and Wang, Rui and Xie, Pengjun and Shen, Ying and Huang, Fei and Zheng, Hai-Tao and Zhang, Rui},
      journal={arXiv preprint arXiv:2103.11647},
      year={2021}
    }





Contact

If you have any questions about this code, please write an e-mail to : xuanjie.wxb@alibaba-inc.com or  dingn18@mails.tsinghua.edu.cn
