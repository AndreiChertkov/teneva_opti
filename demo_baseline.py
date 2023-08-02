from teneva_bm import *
from teneva_opti import *


TASKS = []


BM_M = 1.E+2


BM_AGENT_STEPS = 250
BM_AGENT_M = 1.E+2


BM_OPTI_SEEDS = [0, 1]


OPTIS = [
    OptiTensProtes,
    OptiTensOptimatt,
    OptiTensTtopt,
    OptiTensOpo,
    OptiTensPso,
    OptiTensNb,
    OptiTensSpsa,
    OptiTensPortfolio,
][:3]


BMS_AGENT = [
    BmAgentAnt,
    BmAgentHuman,
    BmAgentHumanStand,
    BmAgentPendInv,
    BmAgentPendInvDouble,
    BmAgentSwimmer,
]


SEEDS = [0, 1]


for d in [10, 20, 50, 80, 100]:
    for Opti in OPTIS:
        for seed in SEEDS:
            TASKS.append({
                'bm': BmQuboKnapDet,
                'bm_args': {'d': d},
                'opti': Opti,
                'opti_args': {'m': BM_M, 'seed': seed},
            })


for Bm in BMS_AGENT:
    for Opti in OPTIS:
        TASKS.append({
            'bm': Bm,
            'bm_args': {'steps': BM_AGENT_STEPS},
            'opti': Opti,
            'opti_args': {'m': BM_AGENT_M},
        })


def demo_baseline():
    oman = OptiManager(TASKS, fold='result_demo_baseline')
    oman.run()


if __name__ == '__main__':
    demo_baseline()
