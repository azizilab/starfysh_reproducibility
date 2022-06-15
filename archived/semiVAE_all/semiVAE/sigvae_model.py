import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.special import softmax
from torch.distributions import Normal, Gamma, Poisson
from torch.distributions import kl_divergence, Distribution


class Arguments:
    """VAE returning arguments"""

    def __init__(self, arg_type, **kwargs):
        assert arg_type == 'encoded' or \
               arg_type == 'decoded' or \
               arg_type == 'latent', \
            "Invalid VAE argument type"

        if arg_type == 'encoded':
            self.x_enc = kwargs['x_enc']
            self.x_enc_samp = kwargs['x_enc_samp'] if 'x_enc_samp' in kwargs.keys() else None

        elif arg_type == 'decoded':
            self.px_mu = kwargs['px_mu']
            self.px_theta = kwargs['px_theta']
            self.px_mu_samp = kwargs['px_mu_samp'] if 'px_mu_samp' in kwargs.keys() else None

        else:  # `latent`
            self.qz_mu = kwargs['qz_mu']
            self.qz_std = kwargs['qz_std']
            self.z = kwargs['z']
            self.qz_mu_samp = kwargs['qz_mu_samp'] if 'qz_mu_samp' in kwargs.keys() else None
            self.qz_std_samp = kwargs['qz_std_samp'] if 'qz_std_samp' in kwargs.keys() else None
            self.z_samp = kwargs['z_samp'] if 'z_samp' in kwargs.keys() else None

    def __iter__(self):
        return self

    def __getitem__(self, k):
        return getattr(self, k)

    def __setitem__(self, k, v):
        return setattr(self, k, v)


# Reference:
# https://github.com/YosefLab/scvi-tools/blob/master/scvi/distributions/_negative_binomial.py
class NegBinom(Distribution):
    """
    Gamma-Poisson mixture approximation of Negative Binomial(mean, dispersion)

    lambda ~ Gamma(mu, theta)
    x ~ Poisson(lambda)
    """

    def __init__(self, mu, theta, eps=1e-5):
        """
        Parameters
        ----------
        mu : torch.Tensor
            mean of NegBinom. distribution
            shape - [# genes,]

        theta : torch.Tensor
            dispersion of NegBinom. distribution
            shape - [# genes,]
        """
        super(NegBinom, self).__init__()
        assert (mu > 0).sum() and (theta > 0).sum(), \
            "Negative mean / dispersion of Negative detected"

        self.mu = mu
        self.theta = theta
        self.eps = eps

    def sample(self):
        lambdas = Gamma(
            concentration=self.theta,
            rate=self.theta / self.mu,
        ).rsample()

        x = Poisson(lambdas).sample()

        return x

    def log_prob(self, x):
        """log-likelihood"""
        ll = torch.lgamma(x + self.theta) - \
             torch.lgamma(x + 1) - \
             torch.lgamma(self.theta) + \
             self.theta * (torch.log(self.theta + self.eps) - torch.log(self.theta + self.mu + self.eps)) + \
             x * (torch.log(self.mu + self.eps) - torch.log(self.theta + self.mu + self.eps))

        return ll


class SignatureVAE(nn.Module):
    """Variational Autoencoder with signature variance priors"""

    def __init__(self,
                 c_in,
                 c_bn,
                 sig_vars_prior,
                 theta_prior=1,
                 n_layers=1,
                 c_hidden_min=64,
                 posterior='Normal'
                 ):
        """
        Parameters
        ----------
        c_in : int
            Num. input features (# input genes)

        c_bn : int
            Num. bottle-neck features

        sog_vars_prior : np.ndarray
            Variance of `signature` spots per `cell_type`
            shape - [#cell_types,]

        theta_prior : float
            Initialized value for posterior variance/dispersion

        n_layers : int
            Num. hidden layers in Encoder/Decoder

        c_hidden_min : int
            Min. features in a single Encoder/Decoder layer

        posterior : str
            Named distribution type for posterior p(x | z)
            Options - 'Normal', 'NegBinom'

        """
        super(SignatureVAE, self).__init__()
        assert posterior == 'Normal' or 'NegBinom', \
            "Unknown posterior distribution type {}, options: (1). Normal; (2).NegBinom".format(posterior)

        # Setting random seed & device
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.n_layers = n_layers
        self.c_in = c_in
        self.c_out = c_in
        self.c_bn = c_bn
        self.posterior = posterior

        sig_vars_prior = torch.Tensor(sig_vars_prior)
        if torch.cuda.is_available() and not sig_vars_prior.is_cuda:
            sig_vars_prior = sig_vars_prior.to(device)

        self.sig_stds = torch.sqrt(sig_vars_prior)
        self.c_hidden = [c_hidden_min]

        for i in range(n_layers - 1):
            c_hidden_min *= 2
            self.c_hidden.append(c_hidden_min)

        self.enc = self.encoder()
        self.dec = self.decoder()

        self.fc_mu_w = nn.Parameter(torch.rand(c_bn, c_bn))
        self.fc_mu_b = nn.Parameter(torch.rand(c_bn))

        # initialize dispersion / std. term for p(x | z)
        log_theta = np.log(theta_prior)
        self.log_theta = nn.Parameter(torch.ones(c_in) * log_theta)

        # initialize parameters for training / loss calculation
        self.libsize = torch.empty(1)
        self.encoded = {}
        self.decoded = {}
        self.latent = {}

    def encoder(self):
        c_enc = self.c_hidden[:self.n_layers][::-1]

        first_layer = self.fc_layer(self.c_in, c_enc[0])
        hidden_layers = self.fc_layers(c_enc)
        last_layer = self.fc_layer(c_enc[-1], self.c_bn, dropout=False, activate=None)

        enc_layers = [first_layer, hidden_layers, last_layer]
        enc_net = nn.Sequential(*enc_layers)
        return enc_net

    def decoder(self):
        c_dec = self.c_hidden[:self.n_layers]
        first_layer = self.fc_layer(self.c_bn, c_dec[0])
        hidden_layers = self.fc_layers(c_dec)

        if self.posterior == 'Normal':
            last_layer = self.fc_layer(c_dec[-1], self.c_out, dropout=False, activate=None)
        else:
            last_layer = self.fc_layer(c_dec[-1], self.c_out, dropout=False, activate='softmax')

        dec_layers = [first_layer, hidden_layers, last_layer]
        dec_net = nn.Sequential(*dec_layers)
        return dec_net

    @staticmethod
    def fc_layer(c_in,
                 c_out,
                 dropout=True,
                 activate='relu',
                 p=0.1):
        assert activate is None or \
               activate == 'relu' or \
               activate == 'softmax', \
            "Invalid activation type, options: (1). None, (2). `relu`, (3). `softmax`"

        fc_net = nn.Sequential(nn.Linear(c_in, c_out))
        if dropout:
            fc_net.add_module('dropout', nn.Dropout(p))
        if activate == 'relu':
            fc_net.add_module('relu', nn.ReLU(inplace=True))
        elif activate == 'softmax':
            fc_net.add_module('softmax', nn.Softmax(dim=-1))

        return fc_net

    def fc_layers(self, c_hidden):
        layers = nn.Sequential()
        for i in range(self.n_layers - 1):
            c_prev = c_hidden[i]
            c_next = c_hidden[i + 1]
            layers.add_module('fc_{}'.format(i + 1), self.fc_layer(c_prev, c_next))

        return layers

    def forward(self, x, x_sample):
        if self.posterior == 'NegBinom':
            x = torch.log1p(x)
            x_sample = torch.log1p(x_sample)

        self.libsize = x.sum(1, keepdim=True)

        # Encoder
        x_enc = self.enc(x)
        x_enc_sample = self.enc(x_sample)

        # Calculate mean & variance/std for Bottle-neck layers
        qz_mu = torch.matmul(x_enc, self.fc_mu_w) + self.fc_mu_b
        qz_mu_sample = torch.matmul(x_enc_sample, self.fc_mu_w) + self.fc_mu_b

        std = self.sig_stds.repeat(qz_mu.shape[0], 1)
        std_sample = self.sig_stds.repeat(qz_mu_sample.shape[0], 1)

        # Sample Bottle-neck layers (z) from q ~ Normal(mu, std)
        q = Normal(qz_mu, std)
        q_sample = Normal(qz_mu_sample, std_sample)

        z = q.rsample()
        z_sample = q_sample.rsample()

        # Decoder
        px_mu = self.dec(z)
        px_mu_sample = self.dec(z_sample)
        px_theta = self.log_theta.exp()

        # Refactor returning values into `encoded`, decoded` & `latent`
        encoded = Arguments(
            arg_type='encoded',
            x_enc=x_enc, x_enc_samp=x_enc_sample
        )
        decoded = Arguments(
            arg_type='decoded',
            px_mu=px_mu, px_mu_samp=px_mu_sample, px_theta=px_theta
        )
        latent = Arguments(
            arg_type='latent',
            qz_mu=qz_mu, qz_mu_samp=qz_mu_sample,
            qz_std=std, qz_std_samp=std_sample,
            z=z, z_samp=z_sample
        )
        self.encoded, self.decoded, self.latent = encoded, decoded, latent

        return encoded, decoded, latent

    def inference(self, x, resample=False):
        """
        One-time forward pass to infer latent variable (z) & reconst. expression matrix (x_hat)
        """
        if self.posterior == 'NegBinom':
            x = torch.log1p(x)

        self.libsize = x.sum(1, keepdim=True)

        # Encoder
        x_enc = self.enc(x)
        qz_mu = torch.matmul(x_enc, self.fc_mu_w) + self.fc_mu_b
        qz_std = self.sig_stds.repeat(qz_mu.shape[0], 1)

        q = Normal(qz_mu, qz_std)
        z = q.rsample()

        # Decoder
        px_mu = self.dec(z)
        px_theta = self.log_theta.exp()

        self.latent = Arguments(
            arg_type='latent',
            qz_mu=qz_mu, qz_std=qz_std, z=z
        )
        latent = self.get_latent()

        self.decoded = Arguments(
            arg_type='decoded',
            px_mu=px_mu, px_theta=px_theta
        )
        x_hat = self.reconstruct(self.decoded, resample=resample)

        return x_hat, latent

    def reconstruct(self, decoded, resample=False):
        """Reconstruct observation (x_hat) from latent space (z)"""
        px_mu = decoded.px_mu
        px_theta = decoded.px_theta
        if self.posterior == 'Normal':
            x_hat = Normal(loc=px_mu, scale=torch.sqrt(px_theta)).rsample() if resample else px_mu
        else:
            x_hat = NegBinom(mu=self.libsize * px_mu, theta=px_theta).sample() if resample else self.libsize * px_mu
        x_hat = x_hat.detach().cpu().numpy()

        return x_hat

    def loss(self,
             x,
             mu_gexp,
             gsva_sig,
             alpha=0.6,
             beta=0.35
             ):
        """
        Calculate loss = (1-α) * loss_full_data + α * loss_anchor + β * loss_sig

            loss_full_data = -elbo = -NLL of data + kl_divergence( q(z|x) || p(z) )
            loss_anchor = MSE(
            loss_sig = BinaryCrossEntropy( qz_mu_sample, gsva_sig )

        """
        if self.posterior == 'NegBinom':
            x = torch.log1p(x)

        # Retrieve parameters for loss calculation
        qz_mu, qz_std = self.latent.qz_mu, self.latent.qz_std
        qz_mu_samp = self.latent.qz_mu_samp
        px_mu, px_theta = self.decoded.px_mu, self.decoded.px_theta

        # Check Dimension consistency
        assert x.shape == px_mu.shape, \
            "Invalid dimension between # genes & posterior distribution params"

        if self.posterior == 'Normal':
            nll = -Normal(loc=px_mu, scale=torch.sqrt(px_theta)).log_prob(x).sum(-1)
        else:  # posterior == 'NegBinom'
            nll = -NegBinom(mu=self.libsize * px_mu, theta=px_theta).log_prob(x).sum(-1)

        q_zx = Normal(qz_mu, qz_std)
        p_z = Normal(torch.zeros_like(qz_mu), torch.ones_like(qz_std))
        kl = kl_divergence(q_zx, p_z).sum(-1)

        loss = (nll + kl).mean()
        loss_anchor = F.mse_loss(qz_mu_samp, mu_gexp)
        loss_sig = F.binary_cross_entropy_with_logits(qz_mu_samp, gsva_sig)

        loss_dict = {
            'nll': nll,
            'kl': kl,
            'anchor': loss_anchor,
            'sig': loss_sig,
            'total': (1-alpha-beta)*loss + alpha*loss_anchor + beta*loss_sig
        }

        return loss_dict

    def get_latent(self):
        """Get latent parameters - qz_mu, qz_std, mu, std"""
        assert type(self.latent) is Arguments, \
            "Please train the model first"

        for attr, val in self.latent.__dict__.items():
            self.latent[attr] = val.detach().cpu().numpy() if torch.is_tensor(val) else val
        return self.latent

    def get_deconvolution(self, thld=1e-2):
        """Get cell-type composition of each spot"""
        assert type(self.latent) is Arguments, \
            "Please train the model first"

        mu = self.latent.qz_mu
        mu = mu.detach().cpu().numpy() if torch.is_tensor(mu) else mu
        mixture = np.apply_along_axis(softmax, axis=1, arr=mu)
        mixture[mixture < thld] = 0
        mixture = np.apply_along_axis(
            lambda x: x / x.sum(),
            axis=1,
            arr=mixture
        )

        return mixture
