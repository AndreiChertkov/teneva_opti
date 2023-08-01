from teneva_bm import *
from teneva_opti import *

M = 1.E+2
STEPS = 250

BMS = [
    BmQuboKnapDet(d=10),
    BmQuboKnapDet(d=20),
    BmQuboKnapDet(d=50),
    BmQuboKnapDet(d=80),
    BmQuboKnapDet(d=100),
    BmAgentAnt(steps=STEPS),
    BmAgentHuman(steps=STEPS),
    BmAgentHumanStand(steps=STEPS),
    BmAgentLake(steps=STEPS),
    BmAgentPendInv(steps=STEPS),
    BmAgentPendInvDouble(steps=STEPS),
    BmAgentSwimmer(steps=STEPS),
]


OPTIS = [
    OptiTensProtes,
    OptiTensOptimatt,
    OptiTensTtopt,
    OptiTensOpo,
    OptiTensPso,
    OptiTensNb,
    OptiTensSpsa,
    OptiTensPortfolio,
]


def demo_baseline():
    oman = OptiManager(BMS, OPTIS, M, fold='result_baseline')
    oman.run()


if __name__ == '__main__':
    demo_baseline()
