from torch.utils.data import DataLoader

from ontology import special_tokens
import logging
import random
import json
from span_config import global_config as cfg
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


class Doc2dialSpanReader():
    def __init__(self, tokenizer, data_path):
        self.nlp = spacy.load('en_core_web_sm')
        if os.path.exists(cfg.data_path+'span_preprocessed_data({}).json'.format(cfg.preprocess_style)):
            self.data = json.load(open(cfg.data_path+'span_preprocessed_data({}).json'.format(cfg.preprocess_style), 'r'))
        else:
            self.data = self.get_all_data(data_path)
            self.preprocess()
            json.dump(self.data, open(cfg.data_path+'span_preprocessed_data({}).json'.format(cfg.preprocess_style), 'w'))
        # tokenizer should support GPTtokenizer's methods
        self.set_stats = self.get_set_stats()
        self.tokenizer = tokenizer
        if cfg.mode == 'train':
            self.add_special_token()
        cfg.pad_id = self.tokenizer.convert_tokens_to_ids('[PAD]') if cfg.PTM == 'bert' else self.tokenizer.convert_tokens_to_ids('<pad>')
        cfg.start_of_user_id = self.tokenizer.convert_tokens_to_ids('<sos_u>')
        cfg.start_of_response_id = self.tokenizer.convert_tokens_to_ids('<sos_r>')
        cfg.end_of_response_id = self.tokenizer.convert_tokens_to_ids('<eos_r>')


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
            dial_dict['doc'] = self.preprocess_sentence(self.data['doc']['doc_data'][doc_type][document]['doc_text'])
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


    def get_data_loader(self, data_name, PTM):
        if PTM == 'xlnet':
            return self.get_xlnet_data_loader(data_name)
        data_list = self.data[data_name]
        # context_component_dict = {'U':'user', 'R':'response','S':'span','P':'paragraph'}
        # context_component = [context_component_dict[symbol] for symbol in cfg.context_scheme ]
        # extra_component = context_component[1:-1]
        data = []
        for dial in data_list:
            turn_list = []


            for turn in dial['turns']:
                if turn['role'] == 'user':
                    turn_list.append(self.tokenizer.encode('<sos_u> ' + turn['utterance'] + ' <eos_u>', add_special_tokens = False))
                elif turn['role'] == 'agent':
                    turn_list.append(self.tokenizer.encode('<sos_r> ' + turn['utterance'] + ' <eos_r>', add_special_tokens = False))


            doc = self.tokenizer.encode(dial['doc'], add_special_tokens = False)
            windows = []
            if len(doc) > cfg.window_size:
                for k in range(math.ceil((len(doc) - cfg.window_size) / cfg.stride_size)):
                    windows.append(doc[k * cfg.stride_size:k * cfg.stride_size + cfg.window_size])
                windows.append(doc[(k+1) * cfg.stride_size:] + [cfg.pad_id]*(cfg.window_size - len(doc) + (k+1) * cfg.stride_size))
                assert len(windows[-1]) == cfg.window_size
            else:
                windows.append(doc+[cfg.pad_id]*(cfg.window_size - len(doc)))
            for i ,turn in enumerate(dial['turns']):
                if turn['role'] == 'agent':
                    # gather context
                    context_list = [turn_list[a] for a in range(i+1) if a >= 0 and a >= i - cfg.context_turn_num]
                    context = []
                    for elem in context_list:
                        context += elem
                    if len(context) > cfg.max_context_len:
                        context = context[-cfg.max_context_len:]
                    context = context + [cfg.pad_id] * (cfg.max_context_len - len(context))

                    span = self.tokenizer.encode(turn['span'], add_special_tokens = False)
                    start_idx = list_index(doc, span)
                    assert start_idx>=0, 'did not find span, check'
                    end_idx = start_idx + len(span) - 1
                    labels = []
                    for k in range(math.ceil(max(len(doc) - cfg.window_size,0) / cfg.stride_size) + 1):
                        label = torch.tensor([cfg.max_context_len + 1, cfg.max_context_len + 1])
                        if start_idx >= k * cfg.stride_size and start_idx < k * cfg.stride_size + cfg.window_size:
                            label[0] = cfg.max_context_len + 2 + start_idx - k * cfg.stride_size
                        if end_idx >= k * cfg.stride_size and end_idx < k * cfg.stride_size + cfg.window_size:
                            label[1] = cfg.max_context_len + 2 + end_idx - k * cfg.stride_size
                        labels.append(label)
                    assert len(labels)
                    # if data_name == 'train':
                    #     for win, lab in zip(windows, labels):
                    #         data.append([torch.tensor(self.tokenizer.build_inputs_with_special_tokens(context, win)), lab])
                    # else:
                    #     data.append([[torch.tensor(self.tokenizer.build_inputs_with_special_tokens(context, win)) for win in windows], labels])
                    if data_name == 'train':
                        if cfg.window_filter_level == 0:
                            for win, lab in zip(windows, labels):
                                data.append([torch.tensor(self.tokenizer.build_inputs_with_special_tokens(context, win)), lab])
                        else:
                            if cfg.window_filter_level == 3:
                                true_win = torch.where(torch.sum((torch.stack(labels) != (cfg.max_context_len + 1)).int(), dim=1) == 2)[0].tolist()
                                picked_win_id = random.randint(0,len(true_win)-1)
                                data.append([torch.tensor(self.tokenizer.build_inputs_with_special_tokens(context, windows[picked_win_id])), labels[picked_win_id]])
                            elif cfg.window_filter_level == 2:
                                not_empty_win = torch.where(torch.sum((torch.stack(labels) != (cfg.max_context_len + 1)).int(), dim=1) != 0)[0].tolist()
                                for picked_win_id in not_empty_win:
                                    data.append([torch.tensor(self.tokenizer.build_inputs_with_special_tokens(context, windows[picked_win_id])), labels[picked_win_id]])
                            elif cfg.window_filter_level == 1:
                                not_empty_win = torch.where(torch.sum((torch.stack(labels) != (cfg.max_context_len + 1)).int(), dim=1) != 0)[0].tolist()
                                empty_win = torch.where(torch.sum((torch.stack(labels) != (cfg.max_context_len + 1)).int(), dim=1) == 0)[0].tolist()
                                try:
                                    picked_empty_win = random.sample(range(len(empty_win), 2))
                                except:
                                    picked_empty_win = empty_win
                                for picked_win_id in not_empty_win + picked_empty_win:
                                    data.append([torch.tensor(self.tokenizer.build_inputs_with_special_tokens(context, windows[picked_win_id])), labels[picked_win_id]])
                    else:
                        data.append([[torch.tensor(self.tokenizer.build_inputs_with_special_tokens(context, win)) for win in windows], labels])

            # logging.info('dial {} done'.format(len(data)))
        self.set_stats[cfg.mode]['steps_per_epoch'] = math.ceil(len(data)/ cfg.batch_size)
        if data_name == 'train':
            return DataLoader(data, batch_size = cfg.batch_size, shuffle = True)
        else:
            return iter(data)
            # return DataLoader(data, batch_size=cfg.batch_size, shuffle = False)

    def get_xlnet_data_loader(self, data_name):
        data_list = self.data[data_name]
        # context_component_dict = {'U':'user', 'R':'response','S':'span','P':'paragraph'}
        # context_component = [context_component_dict[symbol] for symbol in cfg.context_scheme ]
        # extra_component = context_component[1:-1]
        data = []
        drop_count = 0.0
        for dial in data_list:
            turn_list = []

            for turn in dial['turns']:
                if turn['role'] == 'user':
                    turn_list.append(self.tokenizer.encode('<sos_u> ' + turn['utterance'] + ' <eos_u>', add_special_tokens = False))
                elif turn['role'] == 'agent':
                    turn_list.append(self.tokenizer.encode('<sos_r> ' + turn['utterance'] + ' <eos_r>', add_special_tokens = False))


            doc = self.tokenizer.encode(dial['doc'], add_special_tokens = False)
            # windows = []
            # if len(doc) > cfg.window_size:
            #     for k in range(math.ceil((len(doc) - cfg.window_size) / cfg.stride_size)):
            #         windows.append(doc[k * cfg.stride_size:k * cfg.stride_size + cfg.window_size])
            #     windows.append(doc[(k+1) * cfg.stride_size:] + [cfg.pad_id]*(cfg.window_size - len(doc) + (k+1) * cfg.stride_size))
            #     assert len(windows[-1]) == cfg.window_size

            for i ,turn in enumerate(dial['turns']):
                if turn['role'] == 'agent':
                    # gather context
                    context_list = [turn_list[a] for a in range(i+1) if a >= 0 and a >= i - 2]
                    context = []
                    for elem in context_list:
                        context += elem


                    span = self.tokenizer.encode(turn['span'], add_special_tokens = False)
                    start_idx = list_index(doc, span)
                    assert start_idx>=0, 'did not find span, check'
                    end_idx = start_idx + len(span) - 1

                    label = torch.tensor([len(context) + 1 + start_idx, len(context) + 1 + end_idx])
                    input = torch.tensor(self.tokenizer.build_inputs_with_special_tokens(context, doc))
                    if input.shape[0] < 3100:
                        data.append([input , label])
                    else:
                        drop_count += 1



            # logging.info('dial {} done'.format(len(data)))
        self.set_stats[cfg.mode]['steps_per_epoch'] = math.ceil(len(data)/ cfg.batch_size)
        logging.info('drop long {} sample, drop rate {:.4f}'.format(drop_count, drop_count/(len(data)+drop_count)))
        if data_name == 'train':
            return DataLoader(data, batch_size = cfg.batch_size, shuffle = True, collate_fn=PadCollate(pad_value = cfg.pad_id, PTM='xlnet'))
        else:
            return DataLoader(data, batch_size = cfg.batch_size, shuffle = False, collate_fn=PadCollate(pad_value = cfg.pad_id, PTM='xlnet'))
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

