import nevergrad as ng


from teneva_opti import OptiTens


DESC = """
    The PSO optimizer. See the repo nevergrad:
    https://github.com/facebookresearch/nevergrad
"""


class OptiTensPso(OptiTens):
    def __init__(self, *args, **kwargs):
        super().__init__('pso', DESC, *args, **kwargs)

    def _optimize(self):
        self._optimize_ng_helper(ng.optimizers.PSO)
