import datasets
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer
from torch.optim import Adam
import torch
import torch.nn as nn
import os, sys
import random
import argparse
import time
import logging
import json
import math
import numpy as np
from span_config import global_config as cfg
from span_data_reader import Doc2dialSpanReader
from tqdm import tqdm





class UBARdoc():
    def __init__(self, device):
        self.device = device
        # self.tokenizer = AutoTokenizer.from_pretrained(cfg.gpt_path, unk_token='<unk>', sep_token='<sep>',pad_token='<pad>',cls_token='<cls>')
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.gpt_path)
        self.reader = Doc2dialSpanReader(self.tokenizer, cfg.data_path)
        self.model = AutoModelForTokenClassification.from_pretrained(cfg.gpt_path, num_labels=2)
        if cfg.mode == 'train':
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.to(device)


    def train(self):
        optimizer, scheduler = self.get_optimizers()
        global_gradient_step = 0
        data_loader = self.reader.get_data_loader(cfg.mode, cfg.PTM)
        set_stats = self.reader.set_stats[cfg.mode]
        #
        logging.info("***** Running training *****")
        logging.info("  Num Training steps(one turn in a batch of dialogs) per epoch = %d",
                     set_stats['steps_per_epoch'])
        logging.info("  Num Turns = %d", set_stats['num_turns'])
        logging.info("  Num Dialogs = %d", set_stats['num_dials'])
        logging.info("  Num Epochs = %d", cfg.epoch_num)
        logging.info("  Batch size  = %d", cfg.batch_size)
        logging.info("  Gradient Accumulation steps = %d",
                     cfg.gradient_accumulation_steps)
        logging.info("  Total optimization steps = %d",
                     math.ceil(set_stats['steps_per_epoch'] * cfg.epoch_num / cfg.gradient_accumulation_steps ))


        self.model.train()
        for epoch in tqdm(range(cfg.epoch_num), desc='epochs'):
            epoch_step = 0
            tr_loss = 0.0
            logged_loss = 0.0
            epoch_start_time = time.time()
            oom_time = 0  # count out of memory times
            too_long_seq_count = 0  # count sequences that surpass max sequence length

            self.model.zero_grad()
            for batch_idx, batch in tqdm(enumerate(data_loader), desc='steps', total=set_stats['steps_per_epoch']):
                if cfg.PTM != 'xlnet' and batch[0].shape[1]  > cfg.max_seq_length:
                    raise RuntimeError('seq len surpass max seq len. check data preprocession')
                try:

                    # batch = torch.tensor(batch, device=self.device)

                    input = batch[0].to(cfg.device)
                    label = batch[1].to(cfg.device)
                    output = self.model(input_ids=input)
                    # loss = output[0]
                    # loss = self.calculate_loss(output, batch) if cfg.PTM == 'GPT2' else self.calculate_BART_loss(output, label)
                    # loss = self.calculate_bert_loss(output[0], label) if cfg.PTM == 'bert' else self.calculate_xlnet_loss(output[0], label)
                    loss = self.calculate_bert_loss(output[0], label) if cfg.loss_type == 'regular' else self.calculate_adaption_loss(output[0], label)
                    loss.backward()
                    tr_loss += loss.item()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), cfg.gradient_clip)
                    epoch_step += 1

                    if epoch_step % cfg.gradient_accumulation_steps == 0 or epoch_step == set_stats['steps_per_epoch']:
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        global_gradient_step += 1
                        if global_gradient_step % cfg.report_interval == 0:
                            report_loss = (tr_loss - logged_loss)/min(epoch_step, cfg.gradient_accumulation_steps*cfg.report_interval)
                            logged_loss = tr_loss
                            logging.info('')
                            logging.info(
                                'Global gradient step: {}, epoch step: {}, interval loss: {:.8f}'.format(
                                    global_gradient_step, epoch_step, report_loss
                                ))


                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        oom_time += 1
                        logging.warning("run out of memory, times: {}, seq_len: {}".format(oom_time, input.shape[1]))
                    else:
                        logging.info(str(exception))
                        raise exception


            logging.info('')
            logging.info('Train epoch time: {:.2f} min, epoch loss: {:.8f}'.format(
                        (time.time() - epoch_start_time) / 60, tr_loss/set_stats['steps_per_epoch']))
            if (epoch+1) % cfg.model_save_interval == 0:
                self.save_model(epoch, tr_loss/epoch_step)


    def calculate_bert_loss(self, lm_logits, labels):

        loss_fct = nn.CrossEntropyLoss(reduction='sum')
        lm_logits = lm_logits.transpose(1,2)
        loss = loss_fct(
            lm_logits.reshape(-1, lm_logits.shape[-1]), labels.view(-1))

        # num_targets = labels.sum().item()
        num_targets = labels.shape[0]*200

        loss /= num_targets
        return loss

    def calculate_adaption_loss(self, lm_logits, labels):
        # will lower the loss of positive label
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        lm_logits = lm_logits.transpose(1,2)
        loss = loss_fct(lm_logits.reshape(-1, lm_logits.shape[-1]), labels.view(-1))
        num_targets = labels.sum(dim=1).repeat_interleave(2)
        loss /= num_targets
        loss = loss.mean()
        return loss


    def get_optimizers(self):
        # Setup the optimizer and the learning rate scheduler.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.lr)
        num_training_steps = self.reader.set_stats['train']['num_dials'] * \
                             cfg.epoch_num // (cfg.gradient_accumulation_steps * cfg.batch_size)
        num_warmup_steps = cfg.warmup_steps if cfg.warmup_steps >= 0 else int(num_training_steps * 0.2)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        return optimizer, scheduler

    def save_model(self, epoch, loss):
        save_path = os.path.join(
            cfg.exp_path, 'epoch{}_trloss{:.4f}'.format(epoch+1, loss))
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        logging.info('Saving model checkpoint to %s', save_path)
        # save gpt2
        self.model.save_pretrained(save_path)
        # save tokenizer
        self.tokenizer.save_pretrained(save_path)

    def validate(self):
        self.model.eval()
        inference_result_path = os.path.join(cfg.eval_load_path, 'inference_result.json')
        if not os.path.exists(inference_result_path):
            data_loader = self.reader.get_data_loader(cfg.mode, cfg.PTM)
            set_stats = self.reader.set_stats[cfg.mode]
            # logging info
            logging.info("***** Running validating *****")
            logging.info("  Num Turns = %d", set_stats['num_turns'])
            logging.info("  Num Dialogs = %d", set_stats['num_dials'])
            logging.info("***** Inferencing *****")
            pre_result = []

            with torch.no_grad():
                self.model.eval()

                right_count = 0
                for batch_idx, batch in tqdm(enumerate(data_loader), desc='turns', total=set_stats['steps_per_epoch']):
                    input = batch[0].to(cfg.device)
                    label = batch[1].to(cfg.device)

                    output = self.model(input)

                    pre = output[0].transpose(1,2).argmax(-1)
                    # pre_index = torch.stack([index_prefix.clone(), gen_seq], 0).to_list()
                    for i in range(pre.shape[0]):
                        right_count += (pre[i] == label[i]).all().int().item()


        score = right_count/(set_stats['steps_per_epoch']*2*cfg.batch_size)
        logging.info('***** Validate Result *****')
        logging.info('  match rate: {:.6f} '.format(score))
        return score

    def validate_bert(self):
        if cfg.window_filter_level:
            return self.validate_bert_with_filter()

        inference_result_path = os.path.join(cfg.eval_load_path, 'inference_result.json')
        if not os.path.exists(inference_result_path):
            data_loader = self.reader.get_data_loader(cfg.mode, cfg.PTM)
            set_stats = self.reader.set_stats[cfg.mode]
            # logging info
            logging.info("***** Running validating *****")
            logging.info("  Num Turns = %d", set_stats['num_turns'])
            logging.info("  Num Dialogs = %d", set_stats['num_dials'])
            logging.info("***** Inferencing *****")
            pre_result = []
            win_info = []

            with torch.no_grad():
                self.model.eval()

                right_count = 0
                window_right_count = 0
                right_window_count = 0
                side_right_count = 0
                side_window_pick_count = 0
                side_empty_count = 0
                full_window_right_count = 0
                side_window_right_count = 0
                result_list = []

                for batch_idx, batch in tqdm(enumerate(data_loader), desc='turns', total=set_stats['steps_per_epoch']*cfg.batch_size):
                    out_logits_list = []
                    # dial_result_dict = {}
                    for b in torch.utils.data.DataLoader(batch[0], batch_size=cfg.eval_batch_size):
                        b = b.to(cfg.device)
                        output = self.model(b)
                        out_logits_list.append(output[0].cpu())

                    out_logits = torch.cat(out_logits_list, dim=0)


                    pre = out_logits.transpose(1,2).argmax(-1)
                    max_logits = out_logits.max(dim=1)[0]*(pre != (cfg.max_context_len+1)).float()
                    top_num = min(cfg.eval_top_pick, pre.shape[0])
                    top_start_idxs = torch.topk(max_logits[:,0], min(top_num,(max_logits[:,0]!=0).int().sum().item()))[1]
                    top_end_idxs = torch.topk(max_logits[:,1], min(top_num,(max_logits[:,1]!=0).int().sum().item()))[1]
                    selected_win = self.find_same_win(top_start_idxs.tolist(), top_end_idxs.tolist())

                    side_window_pick = 0
                    exact_match = 0
                    side_right = 0
                    if selected_win == -1:
                        selected_win = max_logits.max(dim=1)[0].argmax().item()
                        side_window_pick_count += 1
                        side_window_pick = 1
                        #selected_win = max_logits[:,1].argmax().item()     # if not find index in the same window, pick the highest rated end index window for higher F1.



                    # pre_index = torch.stack([index_prefix.clone(), gen_seq], 0).to_list()
                    true_win = torch.where(torch.sum((torch.stack(batch[1]) != (cfg.max_context_len+1)).int(), dim=1) == 2)[0].tolist()
                    win_info.append([selected_win, true_win])
                    if selected_win in true_win:
                        right_window_count += 1
                        exact_match = (pre[selected_win] == batch[1][selected_win]).all().int().item()
                        right_count += exact_match
                        window_right_count += exact_match
                        side_right = (pre[selected_win] == batch[1][selected_win]).int().max().item()
                        side_right_count += side_right
                        if side_right:
                            side_empty_count += (pre[selected_win] == cfg.max_context_len+1).int().max().item()
                    else:
                        window_right_count += max([(pre[idx] == batch[1][idx]).all().int().item() for idx in true_win])
                    # dial_result_dict['out_logits'] = out_logits.tolist()
                    # dial_result_dict['label'] = [x.tolist() for x in batch[1]]
                    # dial_result_dict['pre'] = pre.tolist()
                    # dial_result_dict['selected_win'] = selected_win
                    # dial_result_dict['true_win'] = true_win
                    # dial_result_dict['full_window_pick'] = 1 - side_window_pick
                    # dial_result_dict['side_right'] = side_right
                    # dial_result_dict['exact_match'] = exact_match
                    # dial_result_dict['win_selection_right'] = int(selected_win in true_win)
                    # dial_result_dict['selected_pre'] = pre[selected_win].tolist()
                    # dial_result_dict['selected_label'] = batch[1][selected_win].tolist()
                    # result_list.append(dial_result_dict)
                    a = 0

        if cfg.generate_win_info:
            json.dump(win_info, open(os.path.join(cfg.data_path, 'win_info.json'), 'w'))
        score = right_count/(set_stats['steps_per_epoch']*cfg.batch_size)
        logging.info('***** Validate Result *****')
        logging.info('  exact match rate: {:.6f} '.format(score))
        logging.info('  win selection right rate: {:.6f} '.format(right_window_count/(set_stats['steps_per_epoch']*cfg.batch_size)))
        logging.info('  true win right rate: {:.6f} '.format(window_right_count/(set_stats['steps_per_epoch']*cfg.batch_size)))
        logging.info('  side right rate: {:.6f} '.format(side_right_count/(set_stats['steps_per_epoch']*cfg.batch_size)))
        logging.info('  another side empty/ non empty rate: {:.6f} / {:.6f}'.format(side_empty_count/side_right_count, 1- side_empty_count/side_right_count))
        logging.info('  full window pick: {:.6f} '.format(1 - side_window_pick_count/(set_stats['steps_per_epoch']*cfg.batch_size)))
        # json.dump(result_list, open(os.path.join(cfg.eval_load_path, 'eval_result.json'),'w'))
        return score

    def validate_bert_with_filter(self):
        inference_result_path = os.path.join(cfg.eval_load_path, 'inference_result.json')
        if not os.path.exists(inference_result_path):
            data_loader = self.reader.get_data_loader(cfg.mode, cfg.PTM)
            set_stats = self.reader.set_stats[cfg.mode]
            win_info = json.load(open(os.path.join(cfg.data_path, 'win_info.json'), 'r'))
            # logging info
            logging.info("***** Running validating *****")
            logging.info("  Num Turns = %d", set_stats['num_turns'])
            logging.info("  Num Dialogs = %d", set_stats['num_dials'])
            logging.info("***** Inferencing *****")
            pre_result = []

            with torch.no_grad():
                self.model.eval()

                right_count = 0
                true_win_right_count = 0
                # window_right_count = 0
                # right_window_count = 0
                side_right_count = 0
                # side_window_pick_count = 0
                # side_empty_count = 0
                # full_window_right_count = 0
                # side_window_right_count = 0
                result_list = []
                for batch_idx, (batch, win_label) in tqdm(enumerate(zip(data_loader, win_info)), desc='turns', total=set_stats['steps_per_epoch'] * cfg.batch_size):

                    # win_label[0] is model predict window_id, win_label[1] is ture window list
                    if win_label[0] in win_label[1]:
                        picked_true_win_id = win_label[0]
                    else:
                        picked_true_win_id = win_label[1][random.randint(0,len(win_label[1])-1)]
                    # dial_result_dict = {}
                    b = torch.stack([batch[0][win_label[0]], batch[0][picked_true_win_id]])
                    label = [batch[1][win_label[0]],batch[1][picked_true_win_id]]
                    b = b.to(cfg.device)
                    output = self.model(b)
                    out_logits = output[0].cpu()


                    pre = out_logits.transpose(1, 2).argmax(-1)
                    if win_label[0] in win_label[1]:
                        exact_match = (pre[0] == label[0]).all().int().item()
                        right_count += exact_match
                        side_right = (pre[0] == label[0]).int().max().item()
                        side_right_count += side_right
                    else:
                        true_win_right_count += (pre[1] == label[1]).all().int().item()





        score = right_count / (set_stats['steps_per_epoch'] * cfg.batch_size)
        logging.info('***** Validate Result *****')
        logging.info('  exact match rate: {:.6f} '.format(score))
        # logging.info('  win selection right rate: {:.6f} '.format(right_window_count / (set_stats['steps_per_epoch'] * cfg.batch_size)))
        # logging.info('  true win right rate: {:.6f} '.format(window_right_count / (set_stats['steps_per_epoch'] * cfg.batch_size)))
        logging.info('  side right rate: {:.6f} '.format(side_right_count / (set_stats['steps_per_epoch'] * cfg.batch_size)))
        logging.info('  golden win right rate: {:.6f} '.format((true_win_right_count + exact_match) / (set_stats['steps_per_epoch'] * cfg.batch_size)))
        # logging.info('  another side empty/ non empty rate: {:.6f} / {:.6f}'.format(side_empty_count / side_right_count, 1 - side_empty_count / side_right_count))
        # logging.info('  full window pick: {:.6f} '.format(1 - side_window_pick_count / (set_stats['steps_per_epoch'] * cfg.batch_size)))
        # json.dump(result_list, open(os.path.join(cfg.eval_load_path, 'eval_result.json'),'w'))
        return score




    def find_same_win(self, a, b):
        for i in a:
            for j in b:
                if i == j:
                    return i
        return -1
    # def to_str(self, sen):
    #     # convert a tensor in GPU to a string
    #     return self.tokenizer.decode(sen if sen[-1].item == 1 else sen[:np.argwhere(sen.cpu().numpy() == 1)[0,0]])
    #
    # def get_last_res(self, sen):
    #     # get the last response sentence from index tensor
    #     return self.to_str(sen[-sen.tolist()[::-1].index(cfg.start_of_response_id):])



def parse_arg_cfg(args):
    # add args to cfg
    if args.cfg:
        for pair in args.cfg:
            k, v = tuple(pair.split('='))
            dtype = type(getattr(cfg, k))  # this is the global cfg comes from config.py
            if dtype == type(None):
                raise ValueError()
            if dtype is bool:
                v = False if v == 'False' else True
            elif dtype is list:
                v = v.split(',')
            else:
                v = dtype(v)
            setattr(cfg, k, v)
    if cfg.PTM == 'xlnet':
        cfg.gpt_path = 'xlnet-base-cased'
    assert cfg.PTM in ['bert','xlnet']
    cfg.stride_size = math.floor(cfg.window_size * cfg.stride_rate)
    return


def main():
    # parse args first
    parser = argparse.ArgumentParser()
    parser.add_argument('-m')
    parser.add_argument('-cfg', nargs='*')
    args = parser.parse_args()

    if args.m:
        cfg.mode = args.m
    parse_arg_cfg(args)

    if cfg.mode == 'validate' or cfg.mode == 'adjust':
        assert(cfg.eval_load_path != 'to be input')
        cfg.gpt_path = cfg.eval_load_path
        if cfg.mode == 'adjust':
            cfg.mode = 'train'
    else:  # train
        if cfg.exp_path in ['', 'to be generated']:

            experiments_path = './span_experiments'
            cfg.exp_path = os.path.join(experiments_path, '{}_sd{}_lr{}_bs{}_ga{}'.format(    cfg.exp_no, cfg.seed,
                                                                                                    cfg.lr, cfg.batch_size,
                                                                                                    cfg.gradient_accumulation_steps))
            if cfg.save_log:
                if not os.path.exists(cfg.exp_path):
                    os.mkdir(cfg.exp_path)

            # to gpt later
            cfg.model_path = os.path.join(cfg.exp_path, 'model.pkl')
            cfg.result_path = os.path.join(cfg.exp_path, 'result.csv')
            cfg.vocab_path_eval = os.path.join(cfg.exp_path, 'vocab')
            cfg.eval_load_path = cfg.exp_path



    cfg._init_logging_handler(cfg.mode)

    #fix random seed
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    #initialize model
    m = UBARdoc(cfg.device)

    if cfg.mode == 'train' :
        m.train()
    elif cfg.mode == 'validate':
        cfg.eval_batch_size = cfg.batch_size
        cfg.batch_size = 1
        m.validate() if cfg.PTM == 'xlnet' else m.validate_bert()

    print('done')

    return 0






if __name__ == '__main__':
    main()
