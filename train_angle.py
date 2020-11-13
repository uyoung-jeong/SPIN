from utils import TrainOptions
from train import AngleTrainer

class Test_Args:
    def __init__(self):
        self.checkpoint = None
        self.dataset = 'h36m-p1'
        self.log_freq = 1000
        self.batch_size = 32
        self.shuffle = False
        self.num_workers = 8
        self.result_file = None

if __name__ == '__main__':
    test_args = Test_Args()
    options = TrainOptions().parse_args()
    trainer = AngleTrainer(options, test_args)
    trainer.train()
