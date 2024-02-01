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
from ax.modelbridge.strategies.alebo import ALEBOStrategy
from ax.modelbridge.strategies.rembo import REMBOStrategy
from ax.modelbridge.strategies.rembo import HeSBOStrategy
import torch
from ax.service.managed_loop import optimize

device = 'cuda' if torch.cuda.is_available() else 'cpu'



sd = 42

random.seed(sd)

D = 513
d = 20
tri = 50
r_init = 5
usr = 1
muu = 0
latent_dims = 2
batch_size = 128

cnt = 0

path1 = "ICML_2024/personalization/bone"
path = os.path.join(path1,f"exp_D{D}_d{d}_tri{tri}_usr{usr}_music{muu}")
os.mkdir(path)

def sc_evaluation_function(parameterization):
    global cnt
    x = np.array(list(parameterization.items()))[:,1].astype(float)
    
    np.savetxt(os.path.join(path,"h.txt"),x)
    score_call()
    score_func = float(input())
    cnt = cnt + 1
    os.rename(os.path.join(path,"h.txt"),os.path.join(path,f"h{cnt}.txt"))
    
    return {"objective": (score_func, 0.0)}

def score_call():
    #np.random.seed(1234)
    bb = random.sample(range(10,25), 9)
    cc = random.sample(range(2,10), 6)
    #bb = []
    dat, fs = load(os.path.join(path1,"test_filts.wav"))
    sig_fft = stft(dat,n_fft=2*(D-1))
    w = np.loadtxt(os.path.join(path,"h.txt"))
    melfb = mel_frequencies(n_mels=D,fmax=fs/2)
    hh = np.interp(np.arange(0,D)/(D-1)*(fs/2), melfb, w)
    r = 10**(hh/10)
    fil = istft((sig_fft.T * r).T)
    wavfile.write(os.path.join(path,f"out{cnt+1}.wav"), fs, fil.astype(dat.dtype))
    display(Audio(fil.astype(dat.dtype),rate=fs))

parameters = [
    {"name": "x0", "type": "range", "bounds": [-10, 10.0], "value_type": "float"},
    {"name": "x1", "type": "range", "bounds": [-10, 10.0], "value_type": "float"},
]
parameters.extend([
    {"name": f"x{i}", "type": "range", "bounds": [-10, 10.0], "value_type": "float"}
    for i in range(2, D)
])

# xd

torch.manual_seed(sd)

hh = np.loadtxt(os.path.join(path1,"xc.txt"))

abo_strategy = ALEBOStrategy(D=D, d=d, init_size=r_init,gp_kwargs={"vae":{},"data":{}})#,device=device
ax_client = AxClient(generation_strategy = abo_strategy, torch_device=device)
ax_client.create_experiment(
    name="sc_evaluation_function",
    parameters = parameters,
    objective_name="objective",
    minimize=True,  
)

par  = []
sco = []
for i in range(5):
    print(f"Running trial {i+1}/5...")
    parameters, trial_index = ax_client.get_next_trial()
    score = sc_evaluation_function(parameters)
    ax_client.complete_trial(trial_index=trial_index, raw_data=score)
    sco.append(score['objective'][0])
    par.append(np.array(list(parameters.items()))[:,1].astype(float))

X_train, y_train =  np.array(par), np.array(sco)


k1 = RBF(length_scale=1e1, length_scale_bounds=(1e-5, 1e5))
gpr = GaussianProcessRegressor(kernel=k1, alpha=1, n_restarts_optimizer = 10)
gpr.fit(X_train, y_train)
opt_kernel_instance=gpr.kernel_
ok1=opt_kernel_instance(X=X_train)

k2 = ExpSineSquared(length_scale=1.0, periodicity=1.0, length_scale_bounds=(1e-5, 1e5))
gpr = GaussianProcessRegressor(kernel=k2, alpha=1, n_restarts_optimizer = 10)
gpr.fit(X_train, y_train)
opt_kernel_instance=gpr.kernel_
ok2=opt_kernel_instance(X=X_train)

k3 = RationalQuadratic(length_scale=1, alpha = 0.1)
gpr = GaussianProcessRegressor(kernel=k3, alpha=1, n_restarts_optimizer = 10)
gpr.fit(X_train, y_train)
opt_kernel_instance=gpr.kernel_
ok3=opt_kernel_instance(X=X_train)

k4 = Matern(length_scale=1e0, length_scale_bounds=(1e-5, 1e5))
gpr = GaussianProcessRegressor(kernel=k4, alpha=1, n_restarts_optimizer = 10)
gpr.fit(X_train, y_train)
opt_kernel_instance=gpr.kernel_
ok4=opt_kernel_instance(X=X_train)

k5 = DotProduct(sigma_0=1e0, sigma_0_bounds=(1e-5, 1e5))
gpr = GaussianProcessRegressor(kernel=k5, alpha=1, n_restarts_optimizer = 10)
gpr.fit(X_train, y_train)
opt_kernel_instance=gpr.kernel_
ok5=opt_kernel_instance(X=X_train)

kkc = []
kkc_norm = []

for i in range(hh.shape[0]):
    kc1 = np.linalg.matrix_power(ok1,int(hh[i,0])) * np.linalg.matrix_power(ok2,int(hh[i,1])) * np.linalg.matrix_power(ok3,int(hh[i,2])) * np.linalg.matrix_power(ok4,int(hh[i,3])) * np.linalg.matrix_power(ok5,int(hh[i,4]))
    kc2 = np.linalg.matrix_power(ok1,int(hh[i,5])) * np.linalg.matrix_power(ok2,int(hh[i,6])) * np.linalg.matrix_power(ok3,int(hh[i,7])) * np.linalg.matrix_power(ok4,int(hh[i,8])) * np.linalg.matrix_power(ok5,int(hh[i,9]))
    kc3 = np.linalg.matrix_power(ok1,int(hh[i,10])) * np.linalg.matrix_power(ok2,int(hh[i,11])) * np.linalg.matrix_power(ok3,int(hh[i,12])) * np.linalg.matrix_power(ok4,int(hh[i,13])) * np.linalg.matrix_power(ok5,int(hh[i,14]))
    kc  = kc1+kc2+kc3
    kkc.append(kc)
    kc_norm = [np.linalg.norm(kc-ok1),np.linalg.norm(kc-ok2),np.linalg.norm(kc-ok3),np.linalg.norm(kc-ok4),np.linalg.norm(kc-ok5)]
    kkc_norm.append(np.array(kc_norm))
np.save(os.path.join(path1,"xd_kc.npy"),kc)
np.save(os.path.join(path1,"xd.npy"),kkc_norm)

# vae

latent_dims = 2
batch_size = 128

class Encoder(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(20, 15)
        self.linear2 = nn.Linear(20, latent_dims)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        return self.linear2(x)

class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 15)
        self.linear2 = nn.Linear(15, 20)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, 1, 20))

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(20, 15)
        self.linear2 = nn.Linear(15, latent_dims)
        self.linear3 = nn.Linear(15, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
def train(autoencoder, data, epochs=50):
    opt = torch.optim.Adam(autoencoder.parameters())
    for epoch in range(epochs):
        for x, y in data:
            x = x.to(device)
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl
            loss.backward()
            opt.step()
    return autoencoder
def plot_latent(autoencoder, data, num_batches=200):
    for i, (x, y) in enumerate(data):
        z = autoencoder.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break
            
xc = np.loadtxt(os.path.join(path1,"xc.txt"))
xd = np.load(os.path.join(path1,"xd.npy"))
dat = np.hstack([xc,xd]).astype('float64')
data = torch.from_numpy(dat.reshape((dat.shape[0],1,dat.shape[1])))
data_y = torch.from_numpy(dat[:,0])

train_dataset = torch.utils.data.TensorDataset(data.float(), data_y)
train_data = torch.utils.data.DataLoader(train_dataset, batch_size, 
                                            shuffle=True)
vae = VariationalAutoencoder(latent_dims).to(device) # GPU
vae = train(vae, train_data)

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

abo_strategy = ALEBOStrategy(D=D, d=d, init_size=r_init,gp_kwargs={"vae":vae,"data":train_data})#,device=device
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
