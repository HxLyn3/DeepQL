from abc import abstractmethod

class BaseRunner:
    """ abstract runner """
    def __init__(self, args):
        if args.env == "atari":
            assert args.backbone == "cnn"
        self.args = args

    @abstractmethod
    def run(self):
        raise NotImplementedError