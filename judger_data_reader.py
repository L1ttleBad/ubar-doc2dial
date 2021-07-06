from torch.utils.data import DataLoader

from ontology import special_tokens
import logging
import random
import json
from judger_config import global_config as cfg
import torch
import spacy
import os
from utils.pad_sequence_collate_fn import PadCollate
import math

def list_index(a, b):
    # find index of list b in list a
    for i in range(len(a)):
        match = 0
        for j in range(len(b)):
            if a[i + j] == b[j]:
                match += 1
            else:
                break
        if j + 1 == match:
            return i
    return -1


class Doc2dialJudgerReader():
    def __init__(self, tokenizer, data_path):
        self.nlp = spacy.load('en_core_web_sm')
        if os.path.exists(cfg.data_path+'judger_preprocessed_data({}).json'.format(cfg.preprocess_style)):
            self.data = json.load(open(cfg.data_path+'judger_preprocessed_data({}).json'.format(cfg.preprocess_style), 'r'))
        else:
            self.data = self.get_all_data(data_path)
            self.preprocess()
            json.dump(self.data, open(cfg.data_path+'judger_preprocessed_data({}).json'.format(cfg.preprocess_style), 'w'))
        # tokenizer should support GPTtokenizer's methods
        self.set_stats = self.get_set_stats()
        self.tokenizer = tokenizer
        if cfg.mode == 'train':
            self.add_special_token()
        cfg.pad_id = self.tokenizer.convert_tokens_to_ids('<pad>')
        cfg.start_of_user_id = self.tokenizer.convert_tokens_to_ids('<sos_u>')
        cfg.start_of_response_id = self.tokenizer.convert_tokens_to_ids('<sos_r>')
        cfg.end_of_response_id = self.tokenizer.convert_tokens_to_ids('<eos_r>')
        cfg.start_of_judger_id = self.tokenizer.convert_tokens_to_ids('<sos_j>')
        cfg.start_of_title_id = self.tokenizer.convert_tokens_to_ids('<sos_t>')
        cfg.end_of_title_id = self.tokenizer.convert_tokens_to_ids('<eos_t>')


        logging.info('Doc2dial reader initialized')


    def add_special_token(self):
        special_tokens_dict = {'additional_special_tokens': special_tokens}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        logging.info('Added special tokens to gpt tokenizer:')
        logging.info(special_tokens_dict)



    def get_all_data(self, data_path):
        data = {}
        try:
            data['train'] = json.load(open(data_path+'doc2dial_dial_train.json', 'r'))
            data['validate'] = json.load(open(data_path+'doc2dial_dial_validation.json', 'r'))
            data['doc'] = json.load(open(data_path+'doc2dial_doc.json', 'r'))
        except:
            logging.error('data file open error, check data path')
        return data

    def preprocess(self):
        for set_name in ['train','validate']:
            text = self.data[set_name]
            dial_list = []
            for doc_type in ['ssa', 'va', 'studentaid', 'dmv']:
                logging.info('start to preprocess {}'.format(doc_type))
                for id, i in enumerate(text['dial_data'][doc_type].items()):
                    dials = i[1]
                    document = i[0]
                    dial_list += self.preprocess_dials(dials, doc_type, document)
                    logging.info('{} docs done!'.format(id+1))
            # self.data[set_name] = dial_list.copy()
            self.data[set_name] = dial_list



    def preprocess_dials(self, dials, doc_type, document):
        dial_list = []
        title_dict = self.get_all_titles(self.data['doc']['doc_data'][doc_type][document])
        for dial in dials:
            dial_dict = {}
            dial_dict['dial_id'] = dial['dial_id']
            dial_dict['doc'] = self.preprocess_sentence(self.data['doc']['doc_data'][doc_type][document]['doc_text'])
            dial_dict['title_dict'] = title_dict
            turns = []
            for turn in dial['turns']:
                ref_span = ''
                if len(turn['references']) > 1:
                    ref_id = [str(i) for i in range(int(turn['references'][0]['sp_id']), int(turn['references'][-1]['sp_id'])+1)]
                else:
                    ref_id = [turn['references'][0]['sp_id']]
                for ref in ref_id:
                    ref_span += self.data['doc']['doc_data'][doc_type][document]['spans'][ref]['text_sp'].strip() + ' '
                ref_paragraph = self.data['doc']['doc_data'][doc_type][document]['spans'][ref]['text_sec'].strip()
                title = self.preprocess_sentence(self.data['doc']['doc_data'][doc_type][document]['spans'][ref]['title'])
                parent_titles = self.data['doc']['doc_data'][doc_type][document]['spans'][ref]['parent_titles']
                if len(parent_titles):
                    title_str = '<sos_h1> ' + self.preprocess_sentence(parent_titles[0]['text']) + ' <eos_h1> <sos_h2> ' + title + ' <eos_h2>'
                else:
                    title_str = '<sos_h1> ' + title + ' <eos_h1>'
                turns.append({'role': turn['role'],
                              'utterance': self.preprocess_sentence(turn['utterance']),
                              'span': self.preprocess_sentence(ref_span[:-1]),
                              'title': title_str,
                              'nt_id': title_dict[title],
                              'paragraph': self.preprocess_sentence(ref_paragraph)})
            dial_dict['turns'] = turns
            # dial_list.append(dial_dict.copy())
            dial_list.append(dial_dict)

        return dial_list

    def get_all_titles(self, doc):
        title_dict = {}
        all_title = []
        near_title = []
        near_title_buf = []

        #count title category num, if there is only one first level title, ignore it.
        start_title_level = 0
        count = [0, 0, 0, 0, 0]
        for id in range(1, len(doc['spans']) + 1):
            span = doc['spans'][str(id)]
            if span['id_sec'][0] == 't':
                count[len(span['parent_titles'])] += 1
        if count[0] == 1:
            start_title_level = 1


        for id in range(1,len(doc['spans'].items())+1):
            span = doc['spans'][str(id)]
            if span['id_sec'][0] == 't' or id == 1:
                title = self.preprocess_sentence(span['title'])
                if not title:
                    title = 'introduction'
                    title_dict[''] = 0
                if len(span['parent_titles']) == start_title_level + 1: #only keep 2 level title
                    if len(span['parent_titles']) > start_title_level + 1:
                        title_dict[title] = len(near_title) - 1
                    all_title.append('<sos_h2> ' + title + ' <eos_h2>')
                    near_title_buf.append('<sos_h2> ' + title + ' <eos_h2>')
                    title_dict[title] = len(near_title) - 1
                else:
                    all_title.append('<sos_h1> ' + title + ' <eos_h1>')
                    near_title.append(' '.join(near_title_buf))
                    near_title_buf = ['<sos_h1> ' + title + ' <eos_h1>']
                    title_dict[title] = len(near_title) - 1
        near_title.append(' '.join(near_title_buf))
        near_title = near_title[1:]
        title_dict['at'] = ' '.join(all_title)
        title_dict['nt'] = near_title
        return title_dict



    def preprocess_sentence(self, sentence):
        return ' '.join([token.norm_ if cfg.preprocess_style == 'norm' else token.text for token in self.nlp(sentence)]).strip()


    def get_data_loader(self, data_name):
        data_list = self.data[data_name]
        # context_component_dict = {'U':'user', 'R':'response','S':'span','P':'paragraph'}
        # context_component = [context_component_dict[symbol] for symbol in cfg.context_scheme ]
        # extra_component = context_component[1:-1]
        data = []
        for dial in data_list:
            dial_list = []
            nt_id_buf = 0


            for turn in dial['turns']:
                if turn['role'] == 'user':
                    dial_list += self.tokenizer.encode('<sos_u> ' + turn['utterance'] + ' <eos_u>', add_special_tokens = False)
                elif turn['role'] == 'agent':
                    dial_list += self.tokenizer.encode('<sos_nt> ' + dial['title_dict']['nt'][turn['nt_id']] + ' <eos_nt>', add_special_tokens = False)
                    dial_list += self.tokenizer.encode('<sos_j> ' + ('<near>' if nt_id_buf == turn['nt_id'] else '<remote>') + ' <eos_j>'  , add_special_tokens = False)
                    dial_list += self.tokenizer.encode('<sos_nt> ' + dial['title_dict']['nt'][turn['nt_id']] + ' <eos_nt>' if nt_id_buf == turn['nt_id'] else '<sos_at> ' + dial['title_dict']['at'] + ' <eos_at>', add_special_tokens=False)
                    dial_list += self.tokenizer.encode('<sos_t> ' + turn['title'] + ' <eos_t>', add_special_tokens = False)
                    nt_id_buf = turn['nt_id']
                    dial_list += self.tokenizer.encode('<sos_r> ' + turn['utterance'] + ' <eos_r>', add_special_tokens = False)


            if len(dial_list) >= cfg.max_seq_length:
                samples = []
                for k in range(math.ceil((len(dial_list) - cfg.max_seq_length) / cfg.stride_size)):
                    samples.append(dial_list[k * cfg.stride_size:k * cfg.stride_size + cfg.max_seq_length])
                samples.append(dial_list[(k + 1) * cfg.stride_size:])
                data += samples
            else:
                data.append(dial_list)
            # logging.info('dial {} done'.format(len(data)))

        data = [torch.tensor(sample_list) for sample_list in data]
        self.set_stats[cfg.mode]['steps_per_epoch'] = math.ceil(len(data)/ cfg.batch_size)
        if data_name == 'train':
            return DataLoader(data, batch_size = cfg.batch_size, shuffle = True, collate_fn=PadCollate(pad_value = cfg.pad_id, PTM=cfg.PTM))
        else:
            return iter(data)
            # return DataLoader(data, batch_size=cfg.batch_size, shuffle = False)




    def get_set_stats(self):
        set_stats = {}
        for set in ['train','validate']:
            data = self.data[set]
            stat_dict = {}
            stat_dict['num_dials'] = len(data)
            num_turns = 0
            for dial in data:
                for _ in dial['turns']:
                    num_turns += 1
            stat_dict['num_turns'] = num_turns
            stat_dict['steps_per_epoch'] = math.ceil(stat_dict['num_dials'] / cfg.batch_size)
            set_stats[set] = stat_dict
        return set_stats

