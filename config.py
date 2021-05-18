import logging, time, os

class _Config:
    def __init__(self):
        self.data_path = './data/'
        self.log_path = './log/'
        self.experiment_path = './experiment/'


        self.save_log = True
        self.log_level = logging.INFO
        self.log_time = time.strftime("%m-%d-%H-%M", time.localtime())

        self.seed = 11

    def _init_logging_handler(self, mode):

        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)

        stderr_handler = logging.StreamHandler()
        if self.save_log:

            file_handler = logging.FileHandler(
                '{}log_{}_{}_sd{}.json'.format(self.log_path,self.log_time, mode,
                                                         self.seed))
            logging.basicConfig(handlers=[stderr_handler, file_handler], level=mode)

            logger = logging.getLogger()
        else:
            logging.basicConfig(handlers=[stderr_handler], level=self.log_level)
        logging.info('logging init finish')
        logging.info('logging level: {}'.format(self.log_level))

global_config = _Config()

