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

import numpy as np
import jax.numpy as jnp
from jax._src.random import PRNGKey
import jax
from typing import Any, Tuple
import functools

from vdm.experiment import Experiment
import vdm.model_vdm_conv as model_vdm_conv
import vdm.model_vdm_base as model_vdm_base


class Experiment_VDM(Experiment):
  """Train and evaluate a VDM model."""

  def get_model_and_params(self, rng: PRNGKey):
    config = self.config
    if config.model.name == "base":
      model_vdm = model_vdm_base
    elif config.model.name == "conv":
      model_vdm = model_vdm_conv

    config = model_vdm.VDMConfig(**config.model)
    model = model_vdm.VDM(config)

    inputs = {"images": jnp.zeros((2, 32, 32, 3), "uint8")}
    inputs["conditioning"] = jnp.zeros((2,))
    inputs["T_eval"] = -jnp.ones((1,))
    rng1, rng2 = jax.random.split(rng)
    params = model.init({"params": rng1, "sample": rng2}, **inputs)
    return model, params

  def loss_fn(self, params, inputs, rng, is_train) -> Tuple[float, Any]:
    rng, sample_rng = jax.random.split(rng)
    rngs = {"sample": sample_rng}
    if is_train:
      rng, dropout_rng = jax.random.split(rng)
      rngs["dropout"] = dropout_rng

    # sample time steps, with antithetic sampling
    outputs = self.state.apply_fn(
        variables={'params': params},
        **inputs,
        rngs=rngs,
        deterministic=not is_train,
        use_t_eval=not is_train,
    )

    rescale_to_bpd = 1./(np.prod(inputs["images"].shape[1:]) * np.log(2.))
    bpd_latent = jnp.mean(outputs.loss_klz) * rescale_to_bpd
    bpd_recon = jnp.mean(outputs.loss_recon) * rescale_to_bpd
    bpd_diff = jnp.mean(outputs.loss_diff) * rescale_to_bpd
    bpd = bpd_recon + bpd_latent + bpd_diff
    scalar_dict = {
        "bpd": bpd,
        "bpd_latent": bpd_latent,
        "bpd_recon": bpd_recon,
        "bpd_diff": bpd_diff,
        "var0": outputs.var_0,
        "var": outputs.var_1,
    }
    img_dict = {"inputs": inputs["images"]}
    metrics = {"scalars": scalar_dict, "images": img_dict}

    return bpd, metrics

  def mse_fn(self, params, inputs, rng, is_train) -> Tuple[float, Any]:
    rng, sample_rng = jax.random.split(rng)
    rngs = {"sample": sample_rng}
    if is_train:
      rng, dropout_rng = jax.random.split(rng)
      rngs["dropout"] = dropout_rng

    # sample time steps, with antithetic sampling
    outputs = self.state.apply_fn(
        variables={'params': params},
        **inputs,
        rngs=rngs,
        deterministic=not is_train,
        use_t_eval=not is_train, 
        method=self.model.calc_mse
    )

    rescale_to_bpd = 1./(np.log(2.))
    # bpd_latent = outputs.loss_klz * rescale_to_bpd
    # bpd_recon = outputs.loss_recon * rescale_to_bpd
    # bpd_diff = outputs.loss_diff * rescale_to_bpd
    # bpd = bpd_recon + bpd_latent + bpd_diff
    scalar_dict = {
        # "bpd": bpd,
        # "bpd_latent": bpd_latent,
        # "bpd_recon": bpd_recon,
        "timesteps": outputs.timesteps,
        "snr_deriv": outputs.snr_deriv,
        "snr_discrete": outputs.snr_discrete,
        "loss_diff": outputs.loss_diff,
        "loss_diff_snr": outputs.loss_diff_snr,
        "f": outputs.f,
        "eps": outputs.eps,
        "eps_hat": outputs.eps_hat,
        "alpha_t": outputs.alpha_t,
        "sigma_t" : outputs.sigma_t,
        # "var0": outputs.var_0,
        # "var": outputs.var_1,
    }
    # img_dict = {"inputs": inputs["images"]}
    # metrics = {"scalars": scalar_dict, "images": img_dict}

    return scalar_dict

  # @functools.partial(jax.jit, static_argnames=['T'])
  def sample_fn(self, *, dummy_inputs, rng, params, T, return_seq=False):
    rng = jax.random.fold_in(rng, jax.lax.axis_index('batch'))
    B, H, W, C = dummy_inputs.shape
    T_ = T.shape[0]
    if return_seq:
      t_eval = T_
    else:
      t_eval = 1
    # if self.model.config.sm_n_timesteps > 0:
    #   T = self.model.config.sm_n_timesteps
    # else:
    #   T = 1000

    conditioning = jnp.zeros((dummy_inputs.shape[0],), dtype='uint8')
    # sample z_0 from the diffusion model
    rng, sample_rng = jax.random.split(rng)
    z_init = jax.random.normal(sample_rng, (t_eval, B, H, W, C))

    def seq_body_fn(i, state):
      z_s = self.state.apply_fn(
          variables={'params': params},
          i=i,
          T=T_,
          z_t=state[i],
          conditioning=conditioning,
          rng=rng,
          method=self.model.sample,
      )
      state = state.at[i+1].set(z_s)
      return state

    def single_body_fn(i, state):
      z_t = state
      z_s = self.state.apply_fn(
          variables={'params': params},
          i=i,
          T=T_,
          z_t=z_t,
          conditioning=conditioning,
          rng=rng,
          method=self.model.sample,
      )
      return z_s

    if return_seq:
      body_fn = seq_body_fn
    else:
      body_fn = single_body_fn

    z_0 = jax.lax.fori_loop(
        lower=0, upper=T_, body_fun=body_fn, init_val=z_init)
    z_0 = jnp.reshape(z_0, (-1, H, W, C))
    
    samples = self.state.apply_fn(
        variables={'params': params},
        z_0=z_0,
        method=self.model.generate_x,
    )

    return samples.reshape(t_eval, B, H, W, C).transpose(1, 0, 2, 3, 4)