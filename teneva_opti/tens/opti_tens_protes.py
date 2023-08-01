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
        conf['_k'] = self._k
        conf['_k_top'] = self._k_top
        conf['_k_gd'] = self._k_gd
        conf['_lr'] = self._lr
        conf['_r'] = self._r
        return conf

    def info(self, footer=''):
        text = ''

        text += '_k (batch size)                          : '
        v = self._k
        text += f'{v}\n'

        text += '_k_top (number of selected candidates)   : '
        v = self._k_top
        text += f'{v}\n'

        text += '_k_gd (number of gradient lifting iters) : '
        v = self._k_gd
        text += f'{v}\n'

        text += '_lr (learning rate for gradient lifting) : '
        v = self._lr
        text += f'{v}\n'

        text += '_r (TT-rank of the inner prob tensor)    : '
        v = self._r
        text += f'{v}\n'

        return super().info(text + footer)

    def opts(self, k=100, k_top=10, k_gd=1, lr=5.E-2, r=5):
        self._k = k
        self._k_top = k_top
        self._k_gd = k_gd
        self._lr = lr
        self._r = r

    def _optimize(self):
        if self.is_n_equal:
            protes(self.target, self.d, self.n0, 1.E+99,
                k=self._k, k_top=self._k_top, k_gd=self._k_gd, lr=self._lr,
                r=self._r, seed=self.seed, is_max=self.is_max)
        else:
            protes_general(self.target, self.n, 1.E+99,
                k=self._k, k_top=self._k_top, k_gd=self._k_gd, lr=self._lr,
                r=self._r, seed=self.seed, is_max=self.is_max)
