import logging, time, os

class _Config:
    def __init__(self):
        self.data_path = './data/'
        self.log_path = './log/'
        self.exp_path = 'to be generated'
        self.eval_load_path = 'to be input'
        self.gpt_path = 'distilgpt2'
        self.exp_no = 'test'
        self.mode = 'train'


        self.save_log = True
        self.report_interval = 100
        self.model_save_interval = 1
        self.log_level = logging.INFO
        self.log_time = time.strftime("%m-%d-%H-%M", time.localtime())

        # training settings
        self.compress_threshold = 0.5
        self.window_size = 300
        self.stride_rate = 0.5
        self.max_context_len = 100
        self.context_turn_num = 2
        self.weight_decay = 0.0
        self.lr = 1e-4
        self.warmup_steps = -1
        self.gradient_accumulation_steps = 8
        self.max_seq_length = 1024
        self.gradient_clip = 5
        self.epoch_num = 5
        self.batch_size = 2  # will be modified to 1 while validating, but will still effect eval_batch_size
        self.seed = 11
        self.device = 0
        self.max_generate_length = 142 # the same as ori paper code. 142 in val and test set



        # Pick context scheme from ['UR', 'USR', 'USPR']
        # self.context_scheme = 'UR'
        self.preprocess_style = 'norm'

        self.PTM = 'GPT2'
        # self.eval_top_pick = 3
        # Pick loss type from ['regular', 'adaption']
        # self.loss_type = 'regular'
        # Pick filter from {0:'use all window', 1: 'filter most empty window', 2: 'filter all empty window', 3:'leave only one true window'}
        # self.window_filter_level = 0
        # self.generate_win_info = False


    def _init_logging_handler(self, mode):

        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)

        stderr_handler = logging.StreamHandler()
        if self.save_log:

            file_handler = logging.FileHandler(
                os.path.join(self.eval_load_path,'log_span_PTM{}_{}_{}.json'.format(self.PTM,self.log_time, mode)))
            logging.basicConfig(handlers=[stderr_handler, file_handler], level=global_config.log_level)

            logger = logging.getLogger()
        else:
            logging.basicConfig(handlers=[stderr_handler], level=self.log_level)
        logging.info('logging init finish')
        logging.info('logging level: {}'.format(self.log_level))
        if self.mode != 'train':
            return
        logging.info('config list:')
        config_attr_list = [i for i in dir(self) if i[0] != '_']
        for attr in config_attr_list:
            logging.info('  {} = {}'.format(attr, getattr(self, attr)))
        logging.info('***************************************************')



global_config = _Config()

