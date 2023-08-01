import nevergrad as ng


from teneva_opti import OptiTens


DESC = """
    The OnePlusOne optimizer. See the repo nevergrad:
    https://github.com/facebookresearch/nevergrad
"""


class OptiTensOpo(OptiTens):
    def __init__(self, *args, **kwargs):
        super().__init__('opo', DESC, *args, **kwargs)

    def _optimize(self):
        self._optimize_ng_helper(ng.optimizers.OnePlusOne)
