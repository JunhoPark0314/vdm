# Copyright 2022 The VDM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Optional, Iterable

import chex
import flax
from flax import linen as nn
import jax
from jax import numpy as jnp
import numpy as np


@flax.struct.dataclass
class VDMConfig:
  """VDM configurations."""
  name: str
  stats: str
  vocab_size: int
  sample_softmax: bool
  antithetic_time_sampling: bool
  with_fourier_features: bool
  with_attention: bool

  # configurations of the noise schedule
  gamma_type: str
  gamma_min: float
  gamma_max: float
  gamma_out: int

  # configurations of the score model
  sm_n_timesteps: int
  sm_n_embd: int
  sm_n_layer: int
  sm_pdrop: float
  sm_kernel_init: Callable = jax.nn.initializers.normal(0.02)
  gamma_shape: tuple = (1,1)




######### Latent VDM model #########
def conv_gamma(gamma, f, square=False):
  if len(f.shape) == 4:
    N,H,W,C = f.shape
    D = 1
    ft = f.transpose(3,1,2,0)
    G = int(N / gamma.shape[0])
  elif len(f.shape) == 5:
    N,H,W,C,D = f.shape
    ft = f.transpose(3,1,2,0,4).reshape(C,H,W,N*D)
    G = int(N*D / gamma.shape[0])
  
  # normalizer = np.sqrt(np.prod(gamma.shape[1:]))
  fft_ft = jnp.fft.rfft2(ft, axes=[1,2])
  if square:
    fft_ft = jnp.square(jnp.absolute(fft_ft))
  ft = jnp.fft.irfft2(fft_ft * jnp.repeat(gamma, G, axis=0).transpose(3,1,2,0), axes=[1, 2])
  # ft = jax.lax.conv_general_dilated(ft, jnp.repeat(gamma, G, axis=0), window_strides=(1,1), padding="SAME", 
  #   feature_group_count=N*D,dimension_numbers=("NHWC", "OIHW", "NHWC"))

  if len(f.shape) == 4:
    ft = ft.transpose(3,1,2,0)
  elif len(f.shape) == 5:
    ft = ft.reshape(C, H, W, N, D).transpose(3,1,2,0,4)
  return ft #/ normalizer



@flax.struct.dataclass
class VDMOutput:
  loss_recon: chex.Array  # [B]
  loss_klz: chex.Array  # [B]
  loss_diff: chex.Array  # [B]
  var_0: float
  var_1: float

@flax.struct.dataclass
class VDMLog:
  loss_diff: chex.Array
  loss_diff_snr : chex.Array
  timesteps: chex.Array
  g_t_grad: chex.Array
  exp_g_t: chex.Array
  snr_discrete: chex.Array
  eps: chex.Array
  eps_hat: chex.Array
  f: chex.Array
  z_t: chex.Array
  alpha_t : chex.Array
  sigma_t : chex.Array
  recon : chex.Array

class VDM(nn.Module):
  config: VDMConfig
  z_std: chex.Array = None

  def setup(self):
    self.encdec = EncDec(self.config)
    self.score_model = ScoreUNet(self.config)
    if self.config.gamma_type == 'learnable_nnet':
      self.gamma = NoiseSchedule_NNet(self.config)
    elif self.config.gamma_type == 'fixed':
      self.gamma = NoiseSchedule_FixedLinear(self.config)
    elif self.config.gamma_type == 'learnable_scalar':
      self.gamma = NoiseSchedule_Scalar(self.config)
    if self.config.gamma_type == 'learnable_fft_nnet':
      self.gamma = NoiseSchedule_FFT_NNet(self.config)
    else:
      raise Exception("Unknown self.var_model")
    

  def sample_eps(self, rng, shape, axes=[1,2]):
    if True:
      real_noise = jax.random.normal(rng, shape=shape)
      imag_noise = jax.random.normal(rng, shape=shape)
      fft_eps = jax.lax.complex(real_noise, imag_noise) * jnp.sqrt(self.z_std * 0.5) #+ self.z_mean
      eps = jnp.fft.irfft2(fft_eps, axes=axes)
    else:
      eps = jax.random.normal(rng, shape=(*shape[:-2], 32, *shape[-1:]))

    return eps

  def __call__(self, images, conditioning, T_eval, deterministic: bool = True, use_t_eval: bool = False):
    g_0, g_1 = self.gamma(0.), self.gamma(1.)
    var_0, var_1 = nn.sigmoid(g_0), nn.sigmoid(g_1)
    mu_0, mu_1 = (1 - jnp.sqrt(1. - var_0)), (1 - jnp.sqrt(1. - var_1))
    x = images
    n_batch = images.shape[0]
    t_eval = T_eval

    # encode
    f = self.encdec.encode(x)

    # 1. RECONSTRUCTION LOSS
    # add noise and reconstruct
    # eps_0 = jax.random.normal(self.make_rng("sample"), shape=f.shape)
    eps_0 = self.sample_eps(self.make_rng("sample"), (f.shape[0],*self.config.gamma_shape,f.shape[-1]))
    # eps_0 = jax.random.normal(self.make_rng("sample"), shape=self.config.gamma_shape)
    # eps_0 = eps_0 * self.config.z_std + self.config.z_mean
    # eps_0 = jnp.fft.rfft2(eps_0, axes=[1,2])

    # z_0 = jnp.sqrt(1. - var_0) * f + jnp.sqrt(var_0) * eps_0
    # z_0_rescaled = f + jnp.exp(0.5 * g_0) * eps_0 # = z_0/sqrt(1-var)
    z_0_rescaled = f + conv_gamma(jnp.exp(0.5 * g_0), eps_0) #+ jnp.fft.irfft2(mu_0 * self.z_mean, axes=[1,2]) # = z_0/sqrt(1-var)
    loss_recon = - self.encdec.logprob(x, z_0_rescaled, g_0)

    # 2. LATENT LOSS
    # KL z1 with N(0,1) prior
    # mean1_sqr = (1. - var_1) * jnp.square(f)

    mean1_sqr = jnp.fft.rfft2(conv_gamma(jnp.sqrt(1. - var_1), f), axes=[1,2]) #+ (mu_1 - 1) * self.z_mean
    mean_diff = (jnp.square(mean1_sqr.real) + jnp.square(mean1_sqr.imag))
    loss_klz = 0.5 * jnp.sum(mean_diff / (self.z_std) + var_1 - jnp.log(var_1) - 1, axis=(1, 2, 3))

    # loss_klz = jnp.fft.ifft2(loss_klz, axes=[1,2])

    # 3. DIFFUSION LOSS
    # sample time steps
    rng1 = self.make_rng("sample")
    if self.config.antithetic_time_sampling:
      t0 = jax.random.uniform(rng1)
      t = jnp.mod(t0 + jnp.arange(0., 1., step=1. / n_batch), 1.)
    else:
      t = jax.random.uniform(rng1, shape=(n_batch,))

    # discretize time steps if we're working with discrete time
    if self.config.sm_n_timesteps > 0:
      T = self.config.sm_n_timesteps
      t = jnp.ceil(t * T) / T
    if use_t_eval:
      T = t_eval
      t = jnp.ceil(t * T) / T

    # sample z_t
    g_t = self.gamma(t)
    # var_t = nn.sigmoid(g_t)[:, None, None, None]
    var_t = nn.sigmoid(g_t)#[:, None, None, None]
    mu_t = (1 - jnp.sqrt(1. - var_t))
    # eps = jax.random.normal(self.make_rng("sample"), shape=f.shape)
    eps = self.sample_eps(self.make_rng("sample"), (f.shape[0],*self.config.gamma_shape,f.shape[-1]))
    # z_t = jnp.sqrt(1. - var_t) * f + jnp.sqrt(var_t) * eps
    z_t = conv_gamma(jnp.sqrt(1. - var_t), f) + conv_gamma(jnp.sqrt(var_t), eps) #+ jnp.fft.irfft2(mu_t * self.z_mean, axes=[1,2])
    # compute predicted noise
    eps_hat = self.score_model(z_t, g_t, conditioning, deterministic)
    # compute MSE of predicted noise
    # loss_diff_mse = jnp.sum(jnp.square(eps - eps_hat), axis=[1, 2, 3])
    loss_diff_mse = jnp.fft.rfft2(eps - eps_hat, axes=[1,2])
    loss_diff_mse = (jnp.square(loss_diff_mse.real) + jnp.square(loss_diff_mse.imag)) / self.z_std


    if (self.config.sm_n_timesteps == 0) and (use_t_eval == False):
      # loss for infinite depth T, i.e. continuous time
      _, g_t_grad = jax.jvp(self.gamma, (t,), (jnp.ones_like(t),))
      loss_diff = .5 * g_t_grad * loss_diff_mse
    else:
      # loss for finite depth T, i.e. discrete time
      s = t - (1./T)
      g_s = self.gamma(s)
      loss_diff = .5 * T * jnp.expm1(g_t - g_s) * loss_diff_mse

    loss_diff = jnp.sum(loss_diff, axis=[1, 2, 3])

    # End of diffusion loss computation

    return VDMOutput(
        loss_recon=loss_recon,
        loss_klz=loss_klz,
        loss_diff=loss_diff,
        var_0=jnp.linalg.norm(var_0) / np.sqrt(self.config.gamma_out),
        var_1=jnp.linalg.norm(var_1) / np.sqrt(self.config.gamma_out),
    )

  def calc_mse(self, images, conditioning, T_eval, deterministic: bool = True, use_t_eval: bool = False):
    g_0, g_1 = self.gamma(0.), self.gamma(1.)
    var_0, var_1 = nn.sigmoid(g_0), nn.sigmoid(g_1)
    mu_0, mu_1 = (1 - jnp.sqrt(1. - var_0)), (1 - jnp.sqrt(1. - var_1))
    x = images
    n_batch = images.shape[0]
    t_eval = T_eval

    # encode
    f = self.encdec.encode(x)

    # 3. DIFFUSION LOSS
    # sample time steps
    rng1 = self.make_rng("sample")
    if self.config.antithetic_time_sampling:
      t0 = jax.random.uniform(rng1)
      t = jnp.mod(t0 + jnp.arange(0., 1., step=1. / n_batch), 1.)
    else:
      t = jax.random.uniform(rng1, shape=(n_batch,))

    # discretize time steps if we're working with discrete time
    if self.config.sm_n_timesteps > 0:
      T = self.config.sm_n_timesteps
      t = jnp.ceil(t * T) / T
    if use_t_eval:
      T = t_eval
      t = jnp.ceil(t * T) / T

    # sample z_t
    g_t = self.gamma(t)
    # var_t = nn.sigmoid(g_t)[:, None, None, None]
    var_t = nn.sigmoid(g_t)#[:, None, None, None]
    mu_t = (1 - jnp.sqrt(1. - var_t))
    # eps = jax.random.normal(self.make_rng("sample"), shape=f.shape)
    eps = self.sample_eps(self.make_rng("sample"), (f.shape[0],*self.config.gamma_shape,f.shape[-1]))
    # z_t = jnp.sqrt(1. - var_t) * f + jnp.sqrt(var_t) * eps
    z_t = conv_gamma(jnp.sqrt(1. - var_t), f) + conv_gamma(jnp.sqrt(var_t), eps) #+ jnp.fft.irfft2(mu_t * self.z_mean, axes=[1,2])
    # compute predicted noise
    eps_hat = self.score_model(z_t, g_t, conditioning, deterministic)
    # compute MSE of predicted noise
    # loss_diff_mse = jnp.sum(jnp.square(eps - eps_hat), axis=[1, 2, 3])
    loss_diff_mse = jnp.fft.rfft2(eps - eps_hat, axes=[1,2])
    loss_diff_mse = (jnp.square(loss_diff_mse.real) + jnp.square(loss_diff_mse.imag)) / self.z_std
    loss_diff_snr = loss_diff_mse * (jnp.sqrt(var_t) / jnp.sqrt(1 - var_t))
    _, g_t_grad = jax.jvp(self.gamma, (t,), (jnp.ones_like(t),))
    s = t - (1./T)
    g_s = self.gamma(s)

    recon = conv_gamma(1/jnp.sqrt(1 - var_t) ,z_t - conv_gamma(jnp.sqrt(var_t), eps_hat))

    return VDMLog(
        loss_diff=loss_diff_mse,
        loss_diff_snr=loss_diff_snr,
        timesteps=t,
        g_t_grad=g_t_grad,
        exp_g_t=jnp.expm1(g_t),
        snr_discrete=jnp.expm1(g_t - g_s),
        eps=conv_gamma(jnp.sqrt(var_t), eps),
        eps_hat=conv_gamma(jnp.sqrt(var_t), eps_hat),
        f=f,
        z_t=z_t,
        alpha_t=jnp.sqrt(1 - var_t),
        sigma_t=jnp.sqrt(var_t),
        recon=recon
    )
  
  def sample(self, i, T, z_t, conditioning, rng):
    rng_body = jax.random.fold_in(rng, i)
    eps = self.sample_eps(rng_body, (z_t.shape[0],*self.config.gamma_shape,z_t.shape[-1]))
    # eps = jax.random.normal(rng_body, z_t.shape)

    t = (T - i) / T
    s = (T - i - 1) / T

    g_s, g_t = self.gamma(s), self.gamma(t)
    var_t , var_s = nn.sigmoid(g_t), nn.sigmoid(g_s)
    sigma_t, sigma_s = jnp.sqrt(var_t), jnp.sqrt(var_s)
    # alpha_t, alpha_s = jnp.sqrt(nn.sigmoid(-g_t)), jnp.sqrt(nn.sigmoid(-g_s))
    # mu_t, mu_s = jnp.sqrt(1 - var_t), jnp.sqrt(1 - var_s)
    alpha_st = jnp.sqrt(nn.sigmoid(-g_s) / nn.sigmoid(-g_t))
    # sigma_ts = (sigma_t ** 2) - (sigma_s / alpha_st) ** 2 
    # mu_ts = jnp.fft.irfft2((1 - 1 / alpha_st) * self.z_mean, axes=[1,2])

    eps_hat = self.score_model(
        z_t,
        # g_t * jnp.ones((z_t.shape[0],), g_t.dtype),
        g_t,
        conditioning,
        deterministic=True)
    a = nn.sigmoid(-g_s)
    # b = nn.sigmoid(-g_t)
    c = - jnp.expm1(g_s - g_t)

    # z_s = jnp.sqrt(nn.sigmoid(-g_s) / nn.sigmoid(-g_t)) * (z_t - sigma_t * c * eps_hat) + \
    #     jnp.sqrt((1. - a) * c) * eps

    z_t_coeff = conv_gamma(alpha_st, z_t)
    eps = conv_gamma(jnp.sqrt((1. - a) * c) ,eps)
    eps_hat = conv_gamma(alpha_st * (-sigma_t * c), eps_hat)
    z_s = z_t_coeff + eps_hat + eps

    return z_s #, z_t_coeff, eps_hat, eps

  def generate_x(self, z_0):
    g_0 = self.gamma(0.)

    var_0 = nn.sigmoid(g_0)
    # z_0_rescaled = z_0 / jnp.sqrt(1. - var_0)
    z_0_rescaled = conv_gamma(1 / jnp.sqrt(1. - var_0), z_0)

    logits = self.encdec.decode(z_0_rescaled, g_0)

    # get output samples
    if self.config.sample_softmax:
      out_rng = self.make_rng("sample")
      samples = jax.random.categorical(out_rng, logits)
    else:
      samples = jnp.argmax(logits, axis=-1)

    return samples

######### Encoder and decoder #########


class EncDec(nn.Module):
  """Encoder and decoder. """
  config: VDMConfig

  def __call__(self, x, g_0):
    # For initialization purposes
    h = self.encode(x)
    return self.decode(h, g_0)

  def encode(self, x):
    # This transforms x from discrete values (0, 1, ...)
    # to the domain (-1,1).
    # Rounding here just a safeguard to ensure the input is discrete
    # (although typically, x is a discrete variable such as uint8)
    x = x.round()
    return 2 * ((x+.5) / self.config.vocab_size) - 1

  def decode(self, z, g_0):
    config = self.config

    # Logits are exact if there are no dependencies between dimensions of x
    x_vals = jnp.arange(0, config.vocab_size)[:, None]
    x_vals = jnp.repeat(x_vals, 3, 1)
    x_vals = self.encode(x_vals).transpose([1, 0])[None, None, None, :, :]
    # inv_stdev = jnp.exp(-0.5 * g_0[..., None])
    # logits = -0.5 * jnp.square((z[..., None] - x_vals) * inv_stdev)
    inv_stdev = jnp.exp(-0.5 * g_0)
    logits = -0.5 * jnp.square(conv_gamma(inv_stdev, (z[..., None] - x_vals)))

    logprobs = jax.nn.log_softmax(logits)
    return logprobs

  def logprob(self, x, z, g_0):
    x = x.round().astype('int32')
    x_onehot = jax.nn.one_hot(x, self.config.vocab_size)
    logprobs = self.decode(z, g_0)
    logprob = jnp.sum(x_onehot * logprobs, axis=(1, 2, 3, 4))
    return logprob


######### Score model #########


class ScoreUNet(nn.Module):
  config: VDMConfig

  @nn.compact
  def __call__(self, z, g_t, conditioning, deterministic=True):
    config = self.config

    # Compute conditioning vector based on 'g_t' and 'conditioning'
    n_embd = self.config.sm_n_embd

    lb = config.gamma_min
    ub = config.gamma_max
    t = (g_t - lb) / (ub - lb)  # ---> [0,1]

    # assert jnp.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1
    # if jnp.isscalar(t):
    #   t = jnp.ones((z.shape[0],), z.dtype) * t
    # elif len(t.shape) == 0:
    #   t = jnp.tile(t[None], z.shape[0])
    G = int(z.shape[0] / t.shape[0])

    temb = get_timestep_embedding(jnp.repeat(t, G, axis=0).reshape(z.shape[0], -1), n_embd)
    cond = jnp.concatenate([temb, conditioning[:, None]], axis=1)
    cond = nn.swish(nn.Dense(features=n_embd * 4, name='dense0')(cond))
    cond = nn.swish(nn.Dense(features=n_embd * 4, name='dense1')(cond))

    # Concatenate Fourier features to input
    if config.with_fourier_features:
      z_f = Base2FourierFeatures(start=6, stop=8, step=1)(z)
      h = jnp.concatenate([z, z_f], axis=-1)
    else:
      h = z

    # Linear projection of input
    h = nn.Conv(features=n_embd, kernel_size=(
        3, 3), strides=(1, 1), name='conv_in')(h)
    hs = [h]

    # Downsampling
    for i_block in range(self.config.sm_n_layer):
      block = ResnetBlock(config, out_ch=n_embd, name=f'down.block_{i_block}')
      h = block(hs[-1], cond, deterministic)[0]
      if config.with_attention:
        h = AttnBlock(num_heads=1, name=f'down.attn_{i_block}')(h)
      hs.append(h)

    # Middle
    h = hs[-1]
    h = ResnetBlock(config, name='mid.block_1')(h, cond, deterministic)[0]
    h = AttnBlock(num_heads=1, name='mid.attn_1')(h)
    h = ResnetBlock(config, name='mid.block_2')(h, cond, deterministic)[0]

    # Upsampling
    for i_block in range(self.config.sm_n_layer + 1):
      b = ResnetBlock(config, out_ch=n_embd, name=f'up.block_{i_block}')
      h = b(jnp.concatenate([h, hs.pop()], axis=-1), cond, deterministic)[0]
      if config.with_attention:
        h = AttnBlock(num_heads=1, name=f'up.attn_{i_block}')(h)

    assert not hs

    # Predict noise
    normalize = nn.normalization.GroupNorm()
    h = nn.swish(normalize(h))
    eps_pred = nn.Conv(
        features=z.shape[-1],
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_init=nn.initializers.zeros,
        name='conv_out')(h)

    # Base measure
    eps_pred += z

    return eps_pred


def get_timestep_embedding(timesteps, embedding_dim: int, dtype=jnp.float32):
  """Build sinusoidal embeddings (from Fairseq).

  This matches the implementation in tensor2tensor, but differs slightly
  from the description in Section 3.5 of "Attention Is All You Need".

  Args:
    timesteps: jnp.ndarray: generate embedding vectors at these timesteps
    embedding_dim: int: dimension of the embeddings to generate
    dtype: data type of the generated embeddings

  Returns:
    embedding vectors with shape `(len(timesteps), embedding_dim)`
  """
  # assert len(timesteps.shape) == 1
  timesteps *= 1000.

  half_dim = embedding_dim // 2
  # emb = np.log(10000) / (half_dim - 1)
  # emb = jnp.exp(jnp.arange(half_dim, dtype=dtype) * -emb)
  # emb = timesteps.astype(dtype)[:, None] * emb[None, :]
  emb = DenseExp(half_dim,name='pos_dense')(timesteps.astype(dtype))
  emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
  if embedding_dim % 2 == 1:  # zero pad
    emb = jax.lax.pad(emb, dtype(0), ((0, 0, 0), (0, 1, 0)))
  assert emb.shape == (timesteps.shape[0], embedding_dim)
  return emb


######### Noise Schedule #########

class NoiseSchedule_Scalar(nn.Module):
  config: VDMConfig

  def setup(self):
    init_bias = self.config.gamma_min
    init_scale = self.config.gamma_max - init_bias
    self.gamma_shape = self.config.gamma_shape
    self.w = self.param('w', constant_init(init_scale), (1,))
    self.b = self.param('b', constant_init(init_bias), (1,))

  @nn.compact
  def __call__(self, t):
    gamma = self.b + abs(self.w) * t
    return gamma.reshape((gamma.shape[0],1)+self.gamma_shape)


class NoiseSchedule_FixedLinear(nn.Module):
  config: VDMConfig

  @nn.compact
  def __call__(self, t):
    config = self.config
    return config.gamma_min + (config.gamma_max-config.gamma_min) * t


class NoiseSchedule_NNet(nn.Module):
  config: VDMConfig
  n_features: int = 1024
  nonlinear: bool = True

  def setup(self):
    config = self.config

    self.gamma_shape = config.gamma_shape
    n_out = int(np.prod(self.gamma_shape))
    kernel_init = nn.initializers.normal()

    init_bias = self.config.gamma_min
    init_scale = self.config.gamma_max - init_bias

    self.l1 = DenseMonotone(n_out,
                            kernel_init=constant_init(init_scale),
                            bias_init=constant_init(init_bias))
    if self.nonlinear:
      self.l2 = DenseMonotone(self.n_features, kernel_init=kernel_init)
      self.l3 = DenseMonotone(n_out, kernel_init=kernel_init, use_bias=False)

  @nn.compact
  def __call__(self, t, det_min_max=False):
    assert jnp.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1

    if jnp.isscalar(t) or len(t.shape) == 0:
      t = t * jnp.ones((1, 1))
    else:
      t = jnp.reshape(t, (-1, 1))

    h = self.l1(t)
    if self.nonlinear:
      _h = 2. * (t - .5)  # scale input to [-1, +1]
      _h = self.l2(_h)
      _h = 2 * (nn.sigmoid(_h) - .5)  # more stable than jnp.tanh(h)
      _h = self.l3(_h) / self.n_features
      h += _h

    # return jnp.squeeze(h, axis=-1)
    return h.reshape((h.shape[0], 1) + self.gamma_shape)

class NoiseSchedule_FFT_NNet(nn.Module):
  config: VDMConfig
  n_features: int = 1024
  nonlinear: bool = True

  def setup(self):
    config = self.config

    self.gamma_shape = config.gamma_shape
    # n_out = int(np.prod(self.gamma_shape))
    n_out = config.gamma_out
    kernel_init = nn.initializers.normal()

    init_bias = self.config.gamma_min
    init_scale = self.config.gamma_max - init_bias

    self.freq = self.param('freq', nn.initializers.uniform(1.0), (1, n_out, 2))
    self.l1 = DenseMonotone(n_out,
                            kernel_init=constant_init(init_scale),
                            bias_init=constant_init(init_bias))
    if self.nonlinear:
      self.l2 = DenseMonotone(self.n_features, kernel_init=kernel_init)
      self.l3 = DenseMonotone(n_out, kernel_init=kernel_init, use_bias=False)
  
  @nn.compact
  def __call__(self, t, det_min_max=False):
    assert jnp.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1

    if jnp.isscalar(t) or len(t.shape) == 0:
      t = t * jnp.ones((1, 1))
    else:
      t = jnp.reshape(t, (-1, 1))

    h = self.l1(t)
    if self.nonlinear:
      _h = 2. * (t - .5)  # scale input to [-1, +1]
      _h = self.l2(_h)
      _h = 2 * (nn.sigmoid(_h) - .5)  # more stable than jnp.tanh(h)
      _h = self.l3(_h) / self.n_features
      h += _h

    return h.reshape((h.shape[0],) + self.gamma_shape + (1,))


def constant_init(value, dtype='float32'):
  def _init(key, shape, dtype=dtype):
    return value * jnp.ones(shape, dtype)
  return _init


class DenseMonotone(nn.Dense):
  """Strictly increasing Dense layer."""

  @nn.compact
  def __call__(self, inputs):
    inputs = jnp.asarray(inputs, self.dtype)
    kernel = self.param('kernel',
                        self.kernel_init,
                        (inputs.shape[-1], self.features))
    kernel = abs(jnp.asarray(kernel, self.dtype))
    y = jax.lax.dot_general(inputs, kernel,
                            (((inputs.ndim - 1,), (0,)), ((), ())),
                            precision=self.precision)
    if self.use_bias:
      bias = self.param('bias', self.bias_init, (self.features,))
      bias = jnp.asarray(bias, self.dtype)
      y = y + bias
    return y

class DenseExp(nn.Dense):
  @nn.compact
  def __call__(self, inputs):
    inputs = jnp.asarray(inputs, self.dtype)
    exp_bias = np.log(10000) / (self.features - 1)
    kernel = self.param('kernel',
                        self.kernel_init,
                        (inputs.shape[-1], self.features))
    # kernel = abs(jnp.asarray(kernel, self.dtype))
    kernel = jnp.exp(jnp.asarray(kernel, self.dtype) * (-exp_bias))
    y = jax.lax.dot_general(inputs, kernel,
                            (((inputs.ndim - 1,), (0,)), ((), ())),
                            precision=self.precision)
    return y

######### ResNet block #########

class ResnetBlock(nn.Module):
  """Convolutional residual block with two convs."""
  config: VDMConfig
  out_ch: Optional[int] = None

  @nn.compact
  def __call__(self, x, cond, deterministic: bool, enc=None):
    config = self.config

    nonlinearity = nn.swish
    normalize1 = nn.normalization.GroupNorm()
    normalize2 = nn.normalization.GroupNorm()

    if enc is not None:
      x = jnp.concatenate([x, enc], axis=-1)

    B, _, _, C = x.shape  # pylint: disable=invalid-name
    out_ch = C if self.out_ch is None else self.out_ch

    h = x
    h = nonlinearity(normalize1(h))
    h = nn.Conv(
        features=out_ch, kernel_size=(3, 3), strides=(1, 1), name='conv1')(h)

    # add in conditioning
    if cond is not None:
      assert cond.shape[0] == B and len(cond.shape) == 2
      h += nn.Dense(
          features=out_ch, use_bias=False, kernel_init=nn.initializers.zeros,
          name='cond_proj')(cond)[:, None, None, :]

    h = nonlinearity(normalize2(h))
    h = nn.Dropout(rate=config.sm_pdrop)(h, deterministic=deterministic)
    h = nn.Conv(
        features=out_ch,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_init=nn.initializers.zeros,
        name='conv2')(h)

    if C != out_ch:
      x = nn.Dense(features=out_ch, name='nin_shortcut')(x)

    assert x.shape == h.shape
    x = x + h
    return x, x


class AttnBlock(nn.Module):
  """Self-attention residual block."""

  num_heads: int

  @nn.compact
  def __call__(self, x):
    B, H, W, C = x.shape  # pylint: disable=invalid-name,unused-variable
    assert C % self.num_heads == 0

    normalize = nn.normalization.GroupNorm()

    h = normalize(x)
    if self.num_heads == 1:
      q = nn.Dense(features=C, name='q')(h)
      k = nn.Dense(features=C, name='k')(h)
      v = nn.Dense(features=C, name='v')(h)
      h = dot_product_attention(
          q[:, :, :, None, :],
          k[:, :, :, None, :],
          v[:, :, :, None, :],
          axis=(1, 2))[:, :, :, 0, :]
      h = nn.Dense(
          features=C, kernel_init=nn.initializers.zeros, name='proj_out')(h)
    else:
      head_dim = C // self.num_heads
      q = nn.DenseGeneral(features=(self.num_heads, head_dim), name='q')(h)
      k = nn.DenseGeneral(features=(self.num_heads, head_dim), name='k')(h)
      v = nn.DenseGeneral(features=(self.num_heads, head_dim), name='v')(h)
      assert q.shape == k.shape == v.shape == (
          B, H, W, self.num_heads, head_dim)
      h = dot_product_attention(q, k, v, axis=(1, 2))
      h = nn.DenseGeneral(
          features=C,
          axis=(-2, -1),
          kernel_init=nn.initializers.zeros,
          name='proj_out')(h)

    assert h.shape == x.shape
    return x + h


def dot_product_attention(query,
                          key,
                          value,
                          dtype=jnp.float32,
                          bias=None,
                          axis=None,
                          # broadcast_dropout=True,
                          # dropout_rng=None,
                          # dropout_rate=0.,
                          # deterministic=False,
                          precision=None):
  """Computes dot-product attention given query, key, and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights. This
  function supports multi-dimensional inputs.


  Args:
    query: queries for calculating attention with shape of `[batch_size, dim1,
      dim2, ..., dimN, num_heads, mem_channels]`.
    key: keys for calculating attention with shape of `[batch_size, dim1, dim2,
      ..., dimN, num_heads, mem_channels]`.
    value: values to be used in attention with shape of `[batch_size, dim1,
      dim2,..., dimN, num_heads, value_channels]`.
    dtype: the dtype of the computation (default: float32)
    bias: bias for the attention weights. This can be used for incorporating
      autoregressive mask, padding mask, proximity bias.
    axis: axises over which the attention is applied.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.

  Returns:
    Output of shape `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]`.
  """
  assert key.shape[:-1] == value.shape[:-1]
  assert (query.shape[0:1] == key.shape[0:1] and
          query.shape[-1] == key.shape[-1])
  assert query.dtype == key.dtype == value.dtype
  input_dtype = query.dtype

  if axis is None:
    axis = tuple(range(1, key.ndim - 2))
  if not isinstance(axis, Iterable):
    axis = (axis,)
  assert key.ndim == query.ndim
  assert key.ndim == value.ndim
  for ax in axis:
    if not (query.ndim >= 3 and 1 <= ax < query.ndim - 2):
      raise ValueError('Attention axis must be between the batch '
                       'axis and the last-two axes.')
  depth = query.shape[-1]
  n = key.ndim
  # batch_dims is  <bs, <non-attention dims>, num_heads>
  batch_dims = tuple(np.delete(range(n), axis + (n - 1,)))
  # q & k -> (bs, <non-attention dims>, num_heads, <attention dims>, channels)
  qk_perm = batch_dims + axis + (n - 1,)
  key = key.transpose(qk_perm)
  query = query.transpose(qk_perm)
  # v -> (bs, <non-attention dims>, num_heads, channels, <attention dims>)
  v_perm = batch_dims + (n - 1,) + axis
  value = value.transpose(v_perm)

  key = key.astype(dtype)
  query = query.astype(dtype) / np.sqrt(depth)
  batch_dims_t = tuple(range(len(batch_dims)))
  attn_weights = jax.lax.dot_general(
      query,
      key, (((n - 1,), (n - 1,)), (batch_dims_t, batch_dims_t)),
      precision=precision)

  # apply attention bias: masking, droput, proximity bias, ect.
  if bias is not None:
    attn_weights = attn_weights + bias

  # normalize the attention weights
  norm_dims = tuple(range(attn_weights.ndim - len(axis), attn_weights.ndim))
  attn_weights = jax.nn.softmax(attn_weights, axis=norm_dims)
  assert attn_weights.dtype == dtype
  attn_weights = attn_weights.astype(input_dtype)

  # compute the new values given the attention weights
  assert attn_weights.dtype == value.dtype
  wv_contracting_dims = (norm_dims, range(value.ndim - len(axis), value.ndim))
  y = jax.lax.dot_general(
      attn_weights,
      value, (wv_contracting_dims, (batch_dims_t, batch_dims_t)),
      precision=precision)

  # back to (bs, dim1, dim2, ..., dimN, num_heads, channels)
  perm_inv = _invert_perm(qk_perm)
  y = y.transpose(perm_inv)
  assert y.dtype == input_dtype
  return y


def _invert_perm(perm):
  perm_inv = [0] * len(perm)
  for i, j in enumerate(perm):
    perm_inv[j] = i
  return tuple(perm_inv)


class Base2FourierFeatures(nn.Module):
  start: int = 0
  stop: int = 8
  step: int = 1

  @nn.compact
  def __call__(self, inputs):
    freqs = range(self.start, self.stop, self.step)

    # Create Base 2 Fourier features
    w = 2.**(jnp.asarray(freqs, dtype=inputs.dtype)) * 2 * jnp.pi
    w = jnp.tile(w[None, :], (1, inputs.shape[-1]))

    # Compute features
    h = jnp.repeat(inputs, len(freqs), axis=-1)
    h = w * h
    h = jnp.concatenate([jnp.sin(h), jnp.cos(h)], axis=-1)
    return h