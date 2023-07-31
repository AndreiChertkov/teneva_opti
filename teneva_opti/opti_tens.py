from teneva_opti import Opti


class OptiTens(Opti):
    def target(self, i):
        return self.target_tens(i)
