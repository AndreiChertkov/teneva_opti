import nevergrad as ng


from teneva_opti import OptiTens


class OptiTensSpsa(OptiTens):
    def __init__(self, *args, **kwargs):
        super().__init__('spsa', *args, **kwargs)

    def _optimize(self):
        self._optimize_ng(ng.optimizers.SPSA)
