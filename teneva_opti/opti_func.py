from teneva_opti import Opti


class OptiFunc(Opti):
    def target(self, x):
        return self.target_func(x)
