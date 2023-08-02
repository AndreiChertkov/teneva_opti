import teneva


from teneva_opti import OptiTens


DESC = """
    The OptimaTT optimizer. See the repo teneva:
    https://github.com/AndreiChertkov/teneva
    and the paper "Optimization of functions given in the tensor train format":
    https://arxiv.org/pdf/2209.14808.pdf
"""


class OptiTensOptimatt(OptiTens):
    def __init__(self, *args, **kwargs):
        super().__init__('optimatt', DESC, *args, **kwargs)

    def get_config(self):
        conf = super().get_config()
        conf['dr_max'] = self.dr_max
        return conf

    def info(self, footer=''):
        text = ''

        text += 'dr_max (max rank increment)              : '
        v = self.dr_max
        text += f'{v}\n'

        return super().info(text + footer)

    def set_opts(self, dr_max=2):
        self.dr_max = dr_max

    def _optimize(self):
        Y = teneva.rand(self.n, r=1)
        Y = teneva.cross(self.target, Y, e=1.E-16, m=self.bm.budget_m-2,
            dr_max=self.dr_max)
        Y = teneva.truncate(Y, e=1.E-16)

        i_min, y_min, i_max, y_max = teneva.optima_tt(Y)
        self.target(i_min)
        self.target(i_max)
