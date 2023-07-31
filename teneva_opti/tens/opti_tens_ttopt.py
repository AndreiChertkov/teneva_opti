from ttopt import TTOpt


from teneva_opti import OptiTens


class OptiTensTtopt(OptiTens):
    def __init__(self, *args, **kwargs):
        """The TTOpt optimizer.

        See the repo ttpot:
        https://github.com/AndreiChertkov/ttopt
        and the paper "TTOpt: A maximum volume quantized tensor train-based
        optimization and its application to reinforcement learning":
        https://openreview.net/forum?id=Kf8sfv0RckB

        """
        super().__init__('ttopt', *args, **kwargs)

    def get_config(self):
        conf = super().get_config()
        conf['_rank'] = self._rank
        conf['_fs_opt'] = self._fs_opt
        return conf

    def opts(self, rank=4, fs_opt=1.):
        self._rank = rank
        self._fs_opt = fs_opt
        return self

    def _optimize(self):
        tto = TTOpt(f=self.target, d=self.d, n=self.n,
            evals=1.E+99, is_func=False, is_vect=True)
        tto.optimize(rank=self._rank, seed=self.seed,
            fs_opt=self._fs_opt, is_max=self.is_max)
