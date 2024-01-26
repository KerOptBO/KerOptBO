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
from random import randrange
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from PIL import Image
import pandas as pd
from __future__ import print_function
from scipy.signal import savgol_filter
import umap
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from six.moves import xrange
import torchvision
from ax.modelbridge.strategies.alebo import ALEBOStrategy
from ax.modelbridge.strategies.rembo import REMBOStrategy
from ax.modelbridge.strategies.rembo import HeSBOStrategy
import torch
from ax.service.managed_loop import optimize

device = 'cuda' if torch.cuda.is_available() else 'cpu'

sd = 42
random.seed(sd)

D = 500
d = 3
tri = 25
r_init = 5
usr = 1
muu = 0
latent_dims = 2
batch_size = 128

cnt = 0

path1 = "ICML_2024/personalization/images"
path = os.path.join(path1,f"exp_D{D}_d{d}_tri{tri}_usr{usr}")
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

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        ### Create an embedding matrix with size number of embedding X embedding dimension
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances between flattened input and embedding vector
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
        
            
        # Choose indices that are min in each row
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        ## Create a matrix of dimensions B*H*W into number of embeddings
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        ### Convert index to on hot encoding 
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
### Create Residual connections
class Residual(nn.Module):
    def __init__(self,in_channels,num_hiddens,num_residual_hiddens):
        super(Residual,self).__init__()
        self._block=nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                     out_channels=num_residual_hiddens,
                     kernel_size=3,stride=1,padding=1,bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                     out_channels=num_hiddens,
                     kernel_size=1,stride=1,bias=False)
        )
        
    def forward(self,x):
        return x + self._block(x)
class ResidualStack(nn.Module):
    def __init__(self,in_channels,num_hiddens,num_residual_layers,num_residual_hiddens):
        super(ResidualStack,self).__init__()
        self._num_residual_layers=num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels,num_hiddens,num_residual_hiddens) for _ in range(self._num_residual_layers)])
    def forward(self,x):
        for i in range(self._num_residual_layers):
            x=self._layers[i](x)
        return F.relu(x)

class Encoder(nn.Module):
    def __init__(self,in_channels,num_hiddens,num_residual_layers,num_residual_hiddens):
        super(Encoder,self).__init__()
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=num_hiddens//2,
                                kernel_size=4,
                                stride=2,padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels = num_hiddens,
                                 kernel_size=4,
                                 stride=2,padding=1
                                )
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                out_channels=num_hiddens,
                                kernel_size=3,
                                stride=1,padding=1)
        self._residual_stack = ResidualStack(in_channels = num_hiddens,
                                             num_hiddens = num_hiddens,
                                             num_residual_layers = num_residual_layers,
                                             num_residual_hiddens = num_residual_hiddens
                                            )
    def forward(self,inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        x = self._conv_2(x)
        x = F.relu(x)
        x = self._conv_3(x)
        return self._residual_stack(x)
class Decoder(nn.Module):
    def __init__(self,in_channels,num_hiddens,num_residual_layers,num_residual_hiddens):
        super(Decoder,self).__init__()
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                out_channels= num_hiddens,
                                kernel_size=3,
                                stride=1,padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens= num_residual_hiddens
                                            )
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                               out_channels=num_hiddens//2,
                                               kernel_size=4,
                                               stride=2,padding=1)
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2,
                                               out_channels=3,
                                               kernel_size=4,
                                               stride=2,padding=1)
    def forward(self,inputs):
        x = self._conv_1(inputs)
        x = self._residual_stack(x)
        x = self._conv_trans_1(x)
        x = F.relu(x)
        return self._conv_trans_2(x)
class Model(nn.Module):
    def __init__(self,num_hiddens,num_residual_layers,num_residual_hiddens,num_embeddings,embedding_dim,commitment_cost,decay=0):
        super(Model,self).__init__()
        self._encoder_= Encoder(3,num_hiddens,num_residual_layers,num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                     out_channels=embedding_dim,
                                     kernel_size=1,
                                     stride=1)
        self._vq_vae = VectorQuantizer(num_embeddings,embedding_dim,commitment_cost)
        self._decoder = Decoder(embedding_dim,
                              num_hiddens,
                              num_residual_layers,
                              num_residual_hiddens)
    def forward(self,x):
        z = self._encoder_(x)
        z = self._pre_vq_conv(z)
        loss,quantized,perplexity,_ = self._vq_vae(z)
        x_recon = self._decoder(quantized)
        return loss,x_recon,perplexity
            
        
num_training_updates = 150
num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 3
embedding_dim= 512
num_embeddings = 512
commitment_cost = 0.25
learning_rate = 3e-4
model = Model(num_hiddens,num_residual_layers,num_residual_hiddens,num_embeddings,embedding_dim,commitment_cost,decay=0)

def load_checkpoint(filepath):
        checkpoint = torch.load(filepath)
        model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])
        for parameter in model.parameters():
            parameter.requires_grad = False

        model.eval()

        return model
def show(img,st):
    npimg = img.numpy()
    plt.figure(figsize = (2,2))
    #fig, = plt.figure(figsize=(2, 2))
    fig = plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.pause(0.5)
    fig.figure.savefig(os.path.join(path,st))
    
model = load_checkpoint("C:/Users/rmirr/Desktop/ICML_2024/Results/checkpoint_file.pth")

train_data_path = "C:/Users/rmirr/Desktop/ICML_2024/Results/gan-getting-started/"
 
### Rescaling incoming image to 28 by 28 pixels
### After Rescaling, convert the image to a tensor
transform = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                               ])
train_data = torchvision.datasets.ImageFolder(root=train_data_path,transform=transform)
train_loader = torch.utils.data.DataLoader(train_data,1,shuffle=True)

# Take a random single batch

for i in range(randrange(200)):
    (valid_originals, _) = next(iter(train_loader))
model.eval()
# vq_output_eval = model._pre_vq_conv(model._encoder_(valid_originals))
# _, valid_quantize, _, _ = model._vq_vae(vq_output_eval)
# for i in range(100):
#         (valid_originals, _) = next(iter(train_loader))

def score_call():
    
    w = np.loadtxt(os.path.join(path,"h.txt"))
#     w = np.repeat(w1, 64, axis=0).reshape((1,512,8,8))
#     vv = torch.tensor(w).float()
#     valid_reconstructions = model._decoder(valid_quantize)
#     show(make_grid(valid_reconstructions.cpu().data+0.5),f"out{cnt+1}.png")
    
#valid_originals = valid_originals.to(device)
    g = int(w[0])
    j = int(w[1])
    m = int(w[2])

    for i in range(randrange(200)):
        (valid_originals, _) = next(iter(train_loader))
    model.eval()
    vq_output_eval = model._pre_vq_conv(model._encoder_(valid_originals))
    _, valid_quantize, _, _ = model._vq_vae(vq_output_eval)
    valid_quantize[:,g,j,m] = valid_quantize[:,g,j,m]#+random.random()
    valid_reconstructions = model._decoder(valid_quantize)
    np.save(os.path.join(path,f"out{cnt+1}.npy"),valid_reconstructions.cpu().data+0.5)
    show(make_grid(valid_reconstructions.cpu().data+0.5),f"out{cnt+1}.png")
    
parameters = [
    {"name": "x0", "type": "range", "bounds": [0, 1511], "value_type": "float"},
    {"name": "x1", "type": "range", "bounds": [0, 7], "value_type": "float"},
    {"name": "x2", "type": "range", "bounds": [0, 7], "value_type": "float"},
]
parameters.extend([
    {"name": f"x{i}", "type": "range", "bounds": [0, 7.0], "value_type": "float"}
    for i in range(3, D)
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
    {"name": "x0", "type": "range", "bounds": [0, 511], "value_type": "float"},
    {"name": "x1", "type": "range", "bounds": [0, 7], "value_type": "float"},
    {"name": "x2", "type": "range", "bounds": [0, 7], "value_type": "float"},
]
parameters.extend([
    {"name": f"x{i}", "type": "range", "bounds": [0, 1], "value_type": "float"}
    for i in range(3, D)
])


torch.manual_seed(sd)
random.seed(sd)

device = torch.device("cuda")# if torch.cuda.is_available() else "cpu")

bp = []
val = []
exp = []
mo = []


torch.manual_seed(sd)

abo_strategy = ALEBOStrategy(D=D, d=d, init_size=r_init,gp_kwargs={"vae":vae,"da":train_data})#,device=device
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
#np.savetxt(os.path.join(path,f"scores.txt"),objectives)
np.savetxt(os.path.join(path,f"scores.txt"),np.hstack([sco,objectives]))
np.savetxt(os.path.join(path,f"h_star.txt"),np.array(list(best_parameters.items()))[:,1].astype(float))
samp = np.array([np.array(list(np.array([trial.arm.parameters for trial in experiment.trials.values()])[i].items()))[:,1].astype(float) for i in range(tri)])
np.savetxt(os.path.join(path,f"samp.txt"),samp)
#acq_val = np.array([experiment.trials[i].generator_run.gen_metadata['expected_acquisition_value'][0] for i in range(r_init,tri)])
#np.savetxt(os.path.join(path,f"acq_val_{ex}.txt"),acq_val)
