import nevergrad as ng


from teneva_opti import OptiTens


class OptiTensPso(OptiTens):
    def __init__(self, *args, **kwargs):
        super().__init__('pso', *args, **kwargs)

    def _optimize(self):
        self._optimize_ng(ng.optimizers.PSO)
