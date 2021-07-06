import datasets
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer, AutoModelForCausalLM
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
from judger_config import global_config as cfg
from judger_data_reader import Doc2dialJudgerReader
from tqdm import tqdm





class UBARdoc():
    def __init__(self, device):
        self.device = device
        # self.tokenizer = AutoTokenizer.from_pretrained(cfg.gpt_path, unk_token='<unk>', sep_token='<sep>',pad_token='<pad>',cls_token='<cls>')
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.gpt_path)
        self.reader = Doc2dialJudgerReader(self.tokenizer, cfg.data_path)
        self.model = AutoModelForCausalLM.from_pretrained(cfg.gpt_path)
        if cfg.mode == 'train':
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.to(device)


    def train(self):
        optimizer, scheduler = self.get_optimizers()
        global_gradient_step = 0
        data_loader = self.reader.get_data_loader(cfg.mode)
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
                if batch[0].shape[0]  > cfg.max_seq_length:
                    raise RuntimeError('seq len surpass max seq len. check data preprocession')
                try:
                    batch = batch.to(self.device)
                    output = self.model(batch)
                    # loss = self.calculate_loss(output, batch) if cfg.PTM == 'GPT2' else self.calculate_BART_loss(output, label)
                    loss = self.calculate_loss(output, batch)
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
                        logging.warning("run out of memory, times: {}, seq_len: {}".format(oom_time, batch.shape[1]))
                    else:
                        logging.info(str(exception))
                        raise exception


            logging.info('')
            logging.info('Train epoch time: {:.2f} min, epoch loss: {:.8f}'.format(
                        (time.time() - epoch_start_time) / 60, tr_loss/set_stats['steps_per_epoch']))
            if (epoch+1) % cfg.model_save_interval == 0:
                self.save_model(epoch, tr_loss/epoch_step)

    def calculate_loss(self, outputs, labels):
        # GPT2-chicahat/train.py
        lm_logits = outputs[0]

        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        pad_id = cfg.pad_id
        loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id, reduction='sum')
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # avg loss
        not_ignore = shift_labels.ne(pad_id)
        num_targets = not_ignore.long().sum().item()

        loss /= num_targets
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

                count = [0,0,0,0,0]
                for batch_idx, batch in tqdm(enumerate(data_loader), desc='turns', total=set_stats['steps_per_epoch']):

                    judger_start_idxs = torch.where(batch == cfg.start_of_judger_id)[0]
                    cut_point = 0
                    for start_idx in judger_start_idxs:
                        count[0] += 1
                        if start_idx - cut_point > 700:
                            cut_point = start_idx - 700
                        input = batch[cut_point: start_idx+1].to(cfg.device)

                        output = self.model.generate(input_ids=input,
                                                 max_length=input.shape[0] + 2,
                                                 temperature=0.7,
                                                 pad_token_id=cfg.pad_id).cpu()

                        if output[-2] != batch[start_idx+1]:
                            continue
                        else:
                            count[1] += 1
                            title_start_idx = start_idx + torch.where(batch[start_idx:] == cfg.start_of_title_id)[0][0]
                            input = batch[cut_point: title_start_idx+1].to(cfg.device)
                            output = self.model.generate(input_ids=input,
                                                         max_length=input.shape[0] + cfg.max_generate_length,
                                                         temperature=0.7,
                                                         pad_token_id=cfg.pad_id,
                                                         eos_token_id=cfg.end_of_title_id).cpu()
                            pre_title = output[title_start_idx-cut_point:]
                            ori_title = batch[title_start_idx:title_start_idx+len(pre_title)]
                            if pre_title == ori_title:
                                count[2] += 1



        score = count[2]/count[0]
        logging.info('***** Validate Result *****')
        logging.info('  match rate: {:.6f} '.format(score))
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
    cfg.stride_size = math.floor(cfg.max_seq_length * cfg.stride_rate)
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

            experiments_path = './judger_experiments'
            cfg.exp_path = os.path.join(experiments_path, '{}_sd{}_bs{}_ga{}'.format(    cfg.exp_no, cfg.seed,
                                                                                                    cfg.batch_size,
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
        cfg.batch_size = 1
        m.validate()

    print('done')

    return 0






if __name__ == '__main__':
    main()
