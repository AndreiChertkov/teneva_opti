import nevergrad as ng


from teneva_opti import OptiTens


class OptiTensOpo(OptiTens):
    def __init__(self, *args, **kwargs):
        super().__init__('opo', *args, **kwargs)

    def _optimize(self):
        self._optimize_ng(ng.optimizers.OnePlusOne)
