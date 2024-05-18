import numpy as np
import math
from ax.utils.measurement.synthetic_functions import branin, hartmann6
import os
from scipy.io import wavfile
from librosa import stft, istft, load, mel_frequencies
from IPython.display import Audio
from IPython.display import display
#from ax.service.utils import report_utils
import shutil
import random
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,Sum, DotProduct, Product, ExpSineSquared, Exponentiation, RationalQuadratic, Matern, PairwiseKernel
import matplotlib.pyplot as plt
from ax.service.ax_client import AxClient
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
from scipy.special import beta
import scipy.stats
from ax.modelbridge.strategies.alebo import ALEBOStrategy
from ax.modelbridge.strategies.rembo import REMBOStrategy
from ax.modelbridge.strategies.rembo import HeSBOStrategy
import torch
from ax.service.managed_loop import optimize

device = 'cuda' if torch.cuda.is_available() else 'cpu'

sd = 42
random.seed(sd)

D = 2000
d = 20
tri = 50
r_init = 5
usr = 1
muu = 0
latent_dims = 2
batch_size = 128

cnt = 0

path1 = "NIPS_2024/synthetic/P1/MCMC"
path = os.path.join(path1,f"exp_D{D}_d{d}_tri{tri}_seed{sd}")
os.mkdir(path)

def sc_evaluation_function(parameterization):
    x = np.array(list(parameterization.items()))[:,1].astype(float)
    score_func = 0
    for i in range(D):
        score_func += math.floor(abs(x[i] + 0.5))**2
    return {"objective": (score_func, 0.0)}

parameters = [
    {"name": "x0", "type": "range", "bounds": [-10, 10.0], "value_type": "float"},
    {"name": "x1", "type": "range", "bounds": [-10, 10.0], "value_type": "float"},
]
parameters.extend([
    {"name": f"x{i}", "type": "range", "bounds": [-10, 10.0], "value_type": "float"}
    for i in range(2, D)
])

# mcmc

class MCMC:
      
    def __init__(self, 
                 dfunc, 
                 chain = 5000, 
                 theta_init = 0.5, 
                 jumpdist = scipy.stats.norm(0,0.2), 
                 space = [-np.inf,np.inf], 
                 burnin = 0, 
                 seed = None) -> None:
        self.dfunc = dfunc
        self.chain = chain
        self.theta_init = theta_init
        self.jump = jumpdist
        self.space = space
        self.burnin = burnin
        self.seed = seed

    def metropolis(self, *arg, **kwarg):
        # update attributes
        for args in arg:
            self.dfunc = args
        
        for kw, value in kwarg.items():
            if kw == 'chain':
                self.chain = value
            elif kw == 'theta_init':
                self.theta_init = value
            elif kw == 'jumpdist':
                self.jump = value
            elif kw == 'space':
                self.space = value
            elif kw == 'burnin':
                self.burnin = value
            elif kw == 'seed':
                self.seed = value
            else:
                raise Exception(f'keyword argument "{kw}" not supported',)

        # check if dfunc callable
        if not callable(self.dfunc):
            raise Exception("dfunc must be a function. recreate the object with a valid density function")
        
        # Metropolis Algorithm
        theta_cur = self.theta_init
        theta_freq = [self.theta_init]
        
        rng = np.random.default_rng(self.seed)

        while True:
            Delta_theta = self.jump.rvs(random_state=rng)
            theta_pro = theta_cur + Delta_theta

            if theta_pro < self.space[0] or theta_pro > self.space[1]:
                pmoving = 0
            elif self.dfunc(theta_cur) == 0:
                pmoving = 1
            else:
                pmoving = min(1,self.dfunc(theta_pro)/self.dfunc(theta_cur))
            
            # np.random.rand()
            if scipy.stats.uniform().rvs(random_state=rng) <= pmoving:
                theta_freq.append(theta_pro)
                theta_cur = theta_pro
            else:
                theta_freq.append(theta_cur)

            if len(theta_freq) >= self.chain:
                break

        return theta_freq[self.burnin:]

parameters = [
    {"name": "x0", "type": "range", "bounds": [-10, 10.0], "value_type": "float"},
    {"name": "x1", "type": "range", "bounds": [-10, 10.0], "value_type": "float"},
]
parameters.extend([
    {"name": f"x{i}", "type": "range", "bounds": [-10, 10.0], "value_type": "float"}
    for i in range(2, D)
])


torch.manual_seed(sd)
random.seed(sd)

device = torch.device("cuda")# if torch.cuda.is_available() else "cpu")

bp = []
val = []
exp = []
mo = []


torch.manual_seed(sd)
density = lambda x: scipy.stats.gamma(2,loc=4,scale=5).pdf(x)
d = MCMC(density, chain = 10, jumpdist=scipy.stats.norm(loc=0,scale=2), space = [0,np.inf])
res = d.metropolis(chain=100, seed = sd)

abo_strategy = ALEBOStrategy(D=D, d=d, init_size=r_init,gp_kwargs={"mcmc":res})#,device=device
print(f"experiment start")
best_parameters, values, experiment, model = optimize(
    parameters=parameters,
    experiment_name="air_513",
    objective_name="objective",
    evaluation_function=sc_evaluation_function,
    minimize=True,
    total_trials=tri,
    random_seed=np.random.seed(sd),
    generation_strategy=abo_strategy,
    #torch_device = device
);
bp.append(best_parameters)
val.append(values)
exp.append(experiment)
mo.append(model)

objectives = np.array([trial.objective_mean for trial in experiment.trials.values()])
np.savetxt(os.path.join(path,f"scores.txt"),objectives)
np.savetxt(os.path.join(path,f"h_star.txt"),np.array(list(best_parameters.items()))[:,1].astype(float))
samp = np.array([np.array(list(np.array([trial.arm.parameters for trial in experiment.trials.values()])[i].items()))[:,1].astype(float) for i in range(tri)])
np.savetxt(os.path.join(path,f"samp.txt"),samp)
#acq_val = np.array([experiment.trials[i].generator_run.gen_metadata['expected_acquisition_value'][0] for i in range(r_init,tri)])
#np.savetxt(os.path.join(path,f"acq_val_{ex}.txt"),acq_val)
