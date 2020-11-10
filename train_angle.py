from utils import TrainOptions
from train import AngleTrainer

if __name__ == '__main__':
    options = TrainOptions().parse_args()
    trainer = AngleTrainer(options)
    trainer.train()
