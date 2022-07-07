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

    if hasattr(self.config.model, 'stats') and self.config.model.stats:
      stats = jnp.load(self.config.model.stats)
      z_std = stats['std']

    config = model_vdm.VDMConfig(**config.model)
    model = model_vdm.VDM(config, z_std)

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
    # rescale_to_bpd = 1./(32 * 17 * 3 * np.log(2.))
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
    scalar_dict = {}
    for k in outputs.__annotations__.keys():
      scalar_dict[k] = getattr(outputs, k)
    # img_dict = {"inputs": inputs["images"]}
    # metrics = {"scalars": scalar_dict, "images": img_dict}

    return scalar_dict

  def sample_intermediate_fn(self, *, dummy_inputs, rng, params, T, return_seq=False):
    rng = jax.random.fold_in(rng, jax.lax.axis_index('batch'))
    B, H, W, C = dummy_inputs.shape
    T_ = T.shape[0]
    B = min(int(1000 / T_), B)
    if return_seq:
      t_eval = T_
    else:
      t_eval = 1
    # if self.model.config.sm_n_timesteps > 0:
    #   T = self.model.config.sm_n_timesteps
    # else:
    #   T = 1000

    conditioning = jnp.zeros((B,), dtype='uint8')
    # sample z_0 from the diffusion model
    rng, sample_rng = jax.random.split(rng)
    if hasattr(self.model, 'sample_eps'):
      z_init = self.model.sample_eps(sample_rng, (t_eval, B, 32, 17, C), axes=[2,3])
    else:
      z_init = jax.random.normal(sample_rng, (t_eval, B, H, W, C))
    
    state = {
      "z_s": z_init,
      "z_t_coeff": jax.random.normal(sample_rng, (t_eval, B, H, W, C)),
      "eps_hat": jax.random.normal(sample_rng, (t_eval, B, H, W, C)),
      "eps": jax.random.normal(sample_rng, (t_eval, B, H, W, C)),
    }

    def seq_body_fn(i, state):
      z_s, z_t_coeff, eps_hat, eps = self.state.apply_fn(
          variables={'params': params},
          i=i,
          T=T_,
          z_t=state["z_s"][i],
          conditioning=conditioning,
          rng=rng,
          method=self.model.sample,
      )
      state["z_s"] = state["z_s"].at[i+1].set(z_s)
      state["z_t_coeff"] = state["z_t_coeff"].at[i+1].set(z_t_coeff)
      state["eps_hat"] = state["eps_hat"].at[i+1].set(eps_hat)
      state["eps"] = state["eps"].at[i+1].set(eps)
      
      return state

    def single_body_fn(i, state):
      z_s, z_t_coeff, eps_hat, eps = self.state.apply_fn(
          variables={'params': params},
          i=i,
          T=T_,
          z_t=state["z_s"][0],
          conditioning=conditioning,
          rng=rng,
          method=self.model.sample,
      )
      state["z_s"] = state["z_s"].at[0].set(z_s)
      return state

    if return_seq:
      body_fn = seq_body_fn
    else:
      body_fn = single_body_fn

    state = jax.lax.fori_loop(
        lower=0, upper=T_, body_fun=body_fn, init_val=state)
    
    for k, v in state.items():
      v = jnp.reshape(v, (-1, H, W, C))
      samples = self.state.apply_fn(
          variables={'params': params},
          z_0=v,
          method=self.model.generate_x,
      )
      state[k] = samples.reshape(t_eval, B, H, W, C).transpose(1, 0, 2, 3, 4)

    return state

  # @functools.partial(jax.jit, static_argnames=['T'])
  def sample_fn(self, *, dummy_inputs, rng, params, T, return_seq=False):
    rng = jax.random.fold_in(rng, jax.lax.axis_index('batch'))
    B, H, W, C = dummy_inputs.shape
    T_ = T.shape[0]
    # B = min(int(1000 / T), B)
    B = min(int(1000 / T_), B)
    if return_seq:
      t_eval = T_
    else:
      t_eval = 1
    # if self.model.config.sm_n_timesteps > 0:
    #   T = self.model.config.sm_n_timesteps
    # else:
    #   T = 1000

    conditioning = jnp.zeros((B,), dtype='uint8')
    # sample z_0 from the diffusion model
    rng, sample_rng = jax.random.split(rng)
    if hasattr(self.model, 'sample_eps'):
      z_init = self.model.sample_eps(sample_rng, (t_eval, B, 32, 17, C), axes=[2,3]) #+ jnp.fft.irfft2(self.model.z_mean, axes=[1,2])[None,:]
    else:
      z_init = jax.random.normal(sample_rng, (t_eval, B, H, W, C))
    
    state = {
      "z_s": z_init,
    }

    def seq_body_fn(i, state):
      z_s = self.state.apply_fn(
          variables={'params': params},
          i=i,
          T=T_,
          z_t=state["z_s"][i],
          conditioning=conditioning,
          rng=rng,
          method=self.model.sample,
      )
      state["z_s"] = state["z_s"].at[i+1].set(z_s)
      
      return state

    def single_body_fn(i, state):
      z_s = self.state.apply_fn(
          variables={'params': params},
          i=i,
          T=T_,
          z_t=state["z_s"][0],
          conditioning=conditioning,
          rng=rng,
          method=self.model.sample,
      )
      state["z_s"] = state["z_s"].at[0].set(z_s)
      return state

    if return_seq:
      body_fn = seq_body_fn
    else:
      body_fn = single_body_fn

    state = jax.lax.fori_loop(
        lower=0, upper=T_, body_fun=body_fn, init_val=state)
    
    for k, v in state.items():
      v = jnp.reshape(v, (-1, H, W, C))
      samples = self.state.apply_fn(
          variables={'params': params},
          z_0=v,
          method=self.model.generate_x,
      )
      state[k] = samples.reshape(t_eval, B, H, W, C).transpose(1, 0, 2, 3, 4)

    return state

