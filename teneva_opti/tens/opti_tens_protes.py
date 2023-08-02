import jax
# jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')
jax.default_device(jax.devices('cpu')[0])


from protes import protes
from protes import protes_general


from teneva_opti import OptiTens


DESC = """
    The PROTES optimizer. See the repo PROTES:
    https://github.com/anabatsh/PROTES
    and the paper "PROTES: Probabilistic optimization with tensor sampling":
    https://arxiv.org/pdf/2301.12162.pdf
"""


class OptiTensProtes(OptiTens):
    def __init__(self, *args, **kwargs):
        super().__init__('protes', DESC, *args, **kwargs)

    def get_config(self):
        conf = super().get_config()
        conf['k'] = self.k
        conf['k_top'] = self.k_top
        conf['k_gd'] = self.k_gd
        conf['lr'] = self.lr
        conf['r'] = self.r
        return conf

    def info(self, footer=''):
        text = ''

        text += 'k (batch size)                           : '
        v = self.k
        text += f'{v}\n'

        text += 'k_top (number of selected candidates)    : '
        v = self.k_top
        text += f'{v}\n'

        text += 'k_gd (number of gradient lifting iters)  : '
        v = self.k_gd
        text += f'{v}\n'

        text += 'lr (learning rate for gradient lifting)  : '
        v = self.lr
        text += f'{v}\n'

        text += 'r (TT-rank of the inner prob tensor)     : '
        v = self.r
        text += f'{v}\n'

        return super().info(text + footer)

    def set_opts(self, k=100, k_top=10, k_gd=1, lr=5.E-2, r=5):
        self.k = k
        self.k_top = k_top
        self.k_gd = k_gd
        self.lr = lr
        self.r = r

    def _optimize(self):
        if self.is_n_equal:
            protes(self.target, self.d, self.n0, 1.E+99,
                k=self.k, k_top=self.k_top, k_gd=self.k_gd, lr=self.lr,
                r=self.r, seed=self.seed, is_max=self.is_max)
        else:
            protes_general(self.target, self.n, 1.E+99,
                k=self.k, k_top=self.k_top, k_gd=self.k_gd, lr=self.lr,
                r=self.r, seed=self.seed, is_max=self.is_max)
