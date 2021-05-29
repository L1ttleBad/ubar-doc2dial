from torch.utils.data import DataLoader

from ontology import special_tokens
import logging
import json
from config import global_config as cfg
import spacy
import os
from utils.pad_sequence_collate_fn import PadCollate
import math

class Doc2dialReader():
    def __init__(self, tokenizer, data_path, context_scheme):
        self.nlp = spacy.load('en_core_web_sm')
        if os.path.exists(cfg.data_path+'preprocessed_data({}).json'.format(cfg.preprocess_style)):
            self.data = json.load(open(cfg.data_path+'preprocessed_data({}).json'.format(cfg.preprocess_style), 'r'))
        else:
            self.data = self.get_all_data(data_path)
            self.preprocess()
            json.dump(self.data, open(cfg.data_path+'preprocessed_data({}).json'.format(cfg.preprocess_style), 'w'))
        # tokenizer should support GPTtokenizer's methods
        self.set_stats = self.get_set_stats()
        self.tokenizer = tokenizer
        if cfg.mode == 'train':
            self.add_special_token()
        cfg.pad_id = self.tokenizer.encode('<pad>')[0]
        cfg.start_of_user_id = self.tokenizer.encode('<sos_u>')[0]
        cfg.start_of_response_id = self.tokenizer.encode('<sos_r>')[0]


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
                for i in text['dial_data'][doc_type].items():
                    dials = i[1]
                    document = i[0]
                    dial_list += self.preprocess_dials(dials, doc_type, document)
            # self.data[set_name] = dial_list.copy()
            self.data[set_name] = dial_list



    def preprocess_dials(self, dials, doc_type, document):
        dial_list = []
        for dial in dials:
            dial_dict = {}
            dial_dict['dial_id'] = dial['dial_id']
            turns = []
            for turn in dial['turns']:
                ref_span = ''
                for ref in turn['references']:
                    ref_span += self.data['doc']['doc_data'][doc_type][document]['spans'][ref['sp_id']]['text_sp'].strip() + ' '
                ref_paragraph = self.data['doc']['doc_data'][doc_type][document]['spans'][ref['sp_id']]['text_sec'].strip()
                turns.append({'role': turn['role'],
                              'utterance': self.preprocess_sentence(turn['utterance']),
                              'span': self.preprocess_sentence(ref_span[:-1]),
                              'paragraph': self.preprocess_sentence(ref_paragraph)})
            dial_dict['turns'] = turns
            # dial_list.append(dial_dict.copy())
            dial_list.append(dial_dict)
        return dial_list

    def preprocess_sentence(self, sentence):
        return ' '.join([token.norm_ if cfg.preprocess_style == 'norm' else token.text for token in self.nlp(sentence)]).strip()


    def get_data_loader(self, data_name):
        data_list = self.data[data_name]
        context_component_dict = {'U':'user', 'R':'response','S':'span','P':'paragraph'}
        context_component = [context_component_dict[symbol] for symbol in cfg.context_scheme ]
        extra_component = context_component[1:-1]
        data = []
        for dial in data_list:
            dial_str = ''

            for turn in dial['turns']:
                if turn['role'] == 'user':
                    dial_str += '<sos_u> ' + turn['utterance'] + ' <eos_u> '
                elif turn['role'] == 'agent':
                    for comp in extra_component:
                        dial_str += '<sos_' + comp[0] + '> ' + turn[comp] + ' <eos_' + comp[0] + '> '
                    dial_str += '<sos_r> ' + turn['utterance'] + ' <eos_r> '
            encoded_string = self.tokenizer.encode(dial_str[:-1])
            total_len = len(encoded_string)
            if not total_len > cfg.max_seq_length:
                data.append(encoded_string)
            else:
                user_start = []
                last_response_start = 0
                while(1):
                    try:
                        last_user_start = encoded_string[last_response_start:].index(cfg.start_of_user_id) + last_response_start
                        user_start.append(last_user_start)
                        last_response_start = encoded_string[last_user_start:].index(cfg.start_of_response_id) + last_user_start
                    except:
                        break

                added = 0
                last_added = -1
                user_start.append(total_len)
                while added < total_len:
                    added = max(added - cfg.max_seq_length//3, 0)
                    if added == last_added: # length surpass limit
                        break
                    for idx in user_start[::-1]:
                        if idx < added + cfg.max_seq_length:
                            data.append(encoded_string[added: idx])
                            #logging.info('seq len: {} cut from: {}-{} total len: {}'.format(idx-added, added, idx, total_len))
                            break
                    last_added = added
                    added = idx
            logging.info('dial {} done'.format(len(data)))
        self.set_stats[cfg.mode]['steps_per_epoch'] = math.ceil(len(data)/ cfg.batch_size)
        return DataLoader(data, batch_size = cfg.batch_size, shuffle = True, collate_fn=PadCollate(pad_value = cfg.pad_id))

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



