import logging, time, os

class _Config:
    def __init__(self):
        self.data_path = './data/'
        self.log_path = './log/'
        self.exp_path = 'to be generated'
        self.gpt_path = 'distilgpt2'
        self.exp_no = 'test'


        self.save_log = True
        self.report_interval = 100
        self.model_save_interval = 3
        self.log_level = logging.INFO
        self.log_time = time.strftime("%m-%d-%H-%M", time.localtime())

        # training settings
        self.lr = 1e-4
        self.warmup_steps = -1
        self.weight_decay = 0.0
        self.gradient_accumulation_steps = 16
        self.max_seq_length = 1024
        self.gradient_clip = 5
        self.epoch_num = 60
        self.batch_size = 2
        self.seed = 11
        self.device = 0


        # Pick context scheme from ['UR', 'USR', 'USPR']
        self.context_scheme = 'UR'
        self.preprocess_style = 'norm'


    def _init_logging_handler(self, mode):

        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)

        stderr_handler = logging.StreamHandler()
        if self.save_log:

            file_handler = logging.FileHandler(
                '{}log_{}_{}_sd{}.json'.format(self.log_path,self.log_time, mode,
                                                         self.seed))
            logging.basicConfig(handlers=[stderr_handler, file_handler], level=global_config.log_level)

            logger = logging.getLogger()
        else:
            logging.basicConfig(handlers=[stderr_handler], level=self.log_level)
        logging.info('logging init finish')
        logging.info('logging level: {}'.format(self.log_level))

global_config = _Config()

