from args import parse_train_opt
from EDGE import EDGE


def train(opt):
    model = EDGE(feature_type = opt.feature_type, checkpoint_path = "./weights/train_checkpoint_gyro_1000.pt")
#    model = EDGE(opt.feature_type)
    model.train_loop(opt)


if __name__ == "__main__":
    opt = parse_train_opt()
    train(opt)
