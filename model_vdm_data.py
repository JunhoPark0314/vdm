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

######### Latent VDM model #########

def normal_pdf(database, images, sigma_t):
  fvar = jnp.square(sigma_t[:, None, None])
  # fvar = jnp.square(sigma_t[:, None, None, None, None]) * 32 * 32 * 0.5
  upper = ((images - database) ** 2 / (2 * fvar)).sum(axis=-1, keepdims=True)
  # fft_img = jnp.fft.rfft2(images.reshape(2, 1, 32, 32, 3), axes=[2,3])
  # fft_data = jnp.fft.rfft2(database.reshape(2, 50000, 32, 32, 3), axes=[2,3])
  # upper = ((jnp.absolute(fft_data - fft_img) ** 2) / (2 * fvar)).sum(axis=[2,3,4])[...,None]

  upper /= (32 * 32 * 3)
  # outer = jnp.log(1 / jnp.sqrt(2 * jnp.pi * fvar))
  return jnp.exp(-upper)

@flax.struct.dataclass
class VDMDataOutput:
  eps_opt: chex.Array
  denoise: chex.Array

class VDM_data(nn.Module):

  def __call__(self, database, sigma_t, alpha_t, images):
    B = images.shape[0]
    f_alpha = alpha_t[:, None, None]
    # fimgs = images.reshape(B, -1)[:, None, :]
    fimgs = (images.reshape(B, -1) * sigma_t[:,None] + database[0,0][None,:] * alpha_t[:,None])[:,None,:]
    pdf = normal_pdf(database * f_alpha, fimgs, sigma_t)
    opt_denoise = (pdf * database).sum(axis=1) / pdf.sum(axis=1)
    return opt_denoise

  def sample(self, database, g_s, g_t, z_t, eps):
    alpha_t, sigma_t = jnp.sqrt(nn.sigmoid(-g_t)), jnp.sqrt(nn.sigmoid(g_t))
    f_alpha = alpha_t[:, None, None]
    # output = self(z_t, alpha_t, sigma_t)

    B = z_t.shape[0]
    fimgs = z_t.reshape(B, -1)[:,None,:]
    pdf = normal_pdf(database * f_alpha, fimgs, sigma_t)
    opt_denoise = (pdf * database).sum(axis=1) / pdf.sum(axis=1)
    opt_denoise = opt_denoise.reshape(-1, 32, 32, 3)
    eps_hat = -(opt_denoise - z_t) / sigma_t[:,None,None,None]

    a = nn.sigmoid(-g_s)
    c = - jnp.expm1(g_s - g_t)
    eps_ = jnp.sqrt((1. - a) * c)[:,None,None,None] * eps
    eps_hat_ = -(jnp.sqrt(nn.sigmoid(-g_s) / nn.sigmoid(-g_t)) * sigma_t * c)[:,None,None,None] * eps_hat
    z_t_coeff = jnp.sqrt(nn.sigmoid(-g_s) / nn.sigmoid(-g_t))[:,None,None,None] * z_t
    z_s = z_t_coeff + eps_ + eps_hat_

    results = {
      "z_s_opt" : z_s,
      "eps_hat_opt": eps_hat,
      "opt_dn": opt_denoise,
    }

    return results