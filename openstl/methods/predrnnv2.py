from .predrnn import PredRNN
from openstl.models import PredRNNv2_Model


class PredRNNv2(PredRNN):

    def __init__(self, **args):
        PredRNN.__init__(self, **args)

    def _build_model(self, **args):
        num_hidden = [int(x) for x in self.hparams.num_hidden.split(',')]
        num_layers = len(num_hidden)
        return PredRNNv2_Model(num_layers, num_hidden, self.hparams)