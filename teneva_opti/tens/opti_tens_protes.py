import jax
# jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')
jax.default_device(jax.devices('cpu')[0]);


from protes import protes
from protes import protes_general


from teneva_opti import OptiTens


class OptiTensProtes(OptiTens):
    def __init__(self, *args, **kwargs):
        """The PROTES optimizer.

        See the repo PROTES:
        https://github.com/anabatsh/PROTES
        and the paper "PROTES: Probabilistic optimization with tensor sampling":
        https://arxiv.org/pdf/2301.12162.pdf

        """
        super().__init__('protes', *args, **kwargs)

    def get_config(self):
        conf = super().get_config()
        conf['_k'] = self._k
        conf['_k_top'] = self._k_top
        conf['_k_gd'] = self._k_gd
        conf['_lr'] = self._lr
        conf['_r'] = self._r
        return conf

    def opts(self, k=100, k_top=10, k_gd=1, lr=5.E-2, r=5):
        self._k = k
        self._k_top = k_top
        self._k_gd = k_gd
        self._lr = lr
        self._r = r
        return self

    def _optimize(self):
        if self.is_n_equal:
            protes(self.target, self.d, self.n0, 1.E+99,
                k=self._k, k_top=self._k_top, k_gd=self._k_gd, lr=self._lr,
                r=self._r, seed=self.seed, is_max=self.is_max)
        else:
            protes_general(self.target, self.n, 1.E+99,
                k=self._k, k_top=self._k_top, k_gd=self._k_gd, lr=self._lr,
                r=self._r, seed=self.seed, is_max=self.is_max)
