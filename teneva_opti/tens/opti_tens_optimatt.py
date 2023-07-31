import teneva


from teneva_opti import OptiTens


class OptiTensOptimatt(OptiTens):
    def __init__(self, *args, **kwargs):
        super().__init__('optimatt', *args, **kwargs)

    def opts(self, dr_max=2):
        self.opts_dr_max = dr_max
        return self

    def _optimize(self):
        Y = teneva.rand(self.n, r=1)
        Y = teneva.cross(self.f_batch, Y, e=1.E-16, m=self.m_max,
            dr_max=self.opts_dr_max)
        Y = teneva.truncate(Y, e=1.E-16)

        i_min, y_min, i_max, y_max = teneva.optima_tt(Y)
        self.f(i_max if self.is_max else i_min)
