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
import vdm.model_vdm_conv_prior as model_vdm_conv_prior
import vdm.model_vdm_conv_xcond as model_vdm_conv_xcond
import vdm.model_vdm_base as model_vdm_base
import vdm.model_vdm_data as model_vdm_data


def get_index():
  y_ar = jnp.tile(jnp.arange(17).reshape(1, 17, 1), (32, 1, 1))
  x_ar = jnp.tile(jnp.arange(32).reshape(32, 1, 1), (1, 17, 1))
  xy = jnp.concatenate([x_ar, y_ar], axis=-1)
  diff = jnp.tile(jnp.square(xy).sum(axis=-1, keepdims=True), (1,1,6)).reshape(-1)
  diff_cut = ((2 ** (jnp.arange(5) + 1)) ** 2)[::-1]
  idx = jnp.arange(32 * 17 * 3 * 2)
  count = jnp.zeros(32 * 17 * 3 * 2)
  idx_list = []
  indices_list = []

  for i in range(5):
    given = idx[diff <= diff_cut[i]]
    trg = idx[diff > diff_cut[i]]
    idx_list.append(jnp.concatenate([given, trg])[None,:])
    indices_list.append(jnp.array([len(given)]))
    count = count.at[trg].set(count[trg] + 1)

  idx = jnp.concatenate(idx_list, axis=0)    
  indices = jnp.concatenate(indices_list, axis=0)

  return idx, indices, count

class Experiment_VDM(Experiment):
  """Train and evaluate a VDM model."""

  def get_data_model(self, train_iter, rng: PRNGKey):
    train_len = 50000
    st = 0
    database = []
    while st < train_len:
      batch = next(train_iter)
      database.append(np.array(batch["images"]).reshape(-1, 32, 32, 3))
      st += database[0].shape[0]
    database = np.vstack(database)[:train_len].reshape(1, train_len, -1)
    database = ((database / 128) - 1)
    data_model = model_vdm_data.VDM_data()
    inputs = {"images": jnp.array(np.random.normal(size=(2,32,32,3))), 
              "database": database,
              "alpha_t": jnp.sqrt(jnp.ones((2,)) * 0.80),
              "sigma_t": jnp.sqrt(jnp.ones((2,)) * 0.20)}
    data_params = data_model.init({"params":rng}, **inputs)
    return data_model, data_params, database

  def get_model_and_params(self, rng: PRNGKey, batch):
    config = self.config
    idx, indices = None, None
    if config.model.name == "base":
      model_vdm = model_vdm_base
    elif config.model.name == "conv":
      model_vdm = model_vdm_conv
    elif config.model.name == "conv_xcond":
      model_vdm = model_vdm_conv_xcond
    elif config.model.name == "conv_prior":
      model_vdm = model_vdm_conv_prior
      # idx, indices, count = get_index()

    if hasattr(self.config.model, 'stats') and self.config.model.stats:
      stats = jnp.load(self.config.model.stats)
      # z_std = stats['std']
      z_std = np.ones_like(stats["std"]) * 32 * 32

    config = model_vdm.VDMConfig(**config.model)
    # model = model_vdm.VDM(config, z_std, idx, count)
    model = model_vdm.VDM(config, z_std)

    # x = (jnp.array(np.random.uniform(size=(100, 32, 32, 3))) * 256).astype(np.uint8)
    inputs = {"images": jnp.array(batch["images"]).reshape(-1, 32, 32, 3)[:100]}
    inputs["conditioning"] = jnp.zeros((inputs["images"].shape[0],))
    inputs["T_eval"] = -jnp.ones((inputs["images"].shape[0],))
    rng1, rng2 = jax.random.split(rng)
    params = model.init({"params": rng1, "sample": rng2}, **inputs)
    return model, params

  def loss_fn(self, params, batch_stat, inputs, rng, is_train) -> Tuple[float, Any]:
    rng, sample_rng = jax.random.split(rng)
    rngs = {"sample": sample_rng}
    if is_train:
      rng, dropout_rng = jax.random.split(rng)
      rngs["dropout"] = dropout_rng
    
    if batch_stat:
      variables={'params': params, 'batch_stats': batch_stat},
      mutable = ['batch_stats']
    else:
      variables={'params': params}
      mutable = False

    # sample time steps, with antithetic sampling
    outputs = self.state.apply_fn(
        variables=variables,
        **inputs,
        rngs=rngs,
        deterministic=not is_train,
        use_t_eval=not is_train,
        mutable=mutable
    )

    if batch_stat:
      outputs, stat_update = outputs
    else:
      stat_update = None

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
        "plog_p": jnp.mean(outputs.plog_p) * rescale_to_bpd,
        "log_q": jnp.mean(outputs.log_q) * rescale_to_bpd,
        "scale": jnp.mean(outputs.scale) * rescale_to_bpd,
    }
    img_dict = {"inputs": inputs["images"]}
    # imd_dict = {"z_t": outputs.z_t,
    #             "grad": outputs.log_snr_t_grad,
    #             "mse_real": outputs.mse_real,
    #             "mse_imag": outputs.mse_imag,
    #             "loss_diff": outputs.loss_diff }

    metrics = {"scalars": scalar_dict, "images": img_dict, "stat_update": stat_update}
    # metrics = {"scalars": scalar_dict, "images": img_dict, "imd": imd_dict}

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
      z_init = self.model.sample_eps(sample_rng, (t_eval, B, 32, 17, C), axes=[2,3], use_var1=True)
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
  def sample_fn(self, *, dummy_inputs, rng, params, batch_stats, T, return_seq=False):
    rng = jax.random.fold_in(rng, jax.lax.axis_index('batch'))
    B, H, W, C = dummy_inputs.shape
    dummy_inputs = (jnp.array(dummy_inputs) - 127.5) / 127.5
    T_ = T.shape[0]
    # B = min(int(1000 / T), B)
    B = min(int(1000 / T_), B)
    # dummy_inputs = dummy_inputs[:B]
    if return_seq:
      t_eval = T_
    else:
      t_eval = 1
    # if self.model.config.sm_n_timesteps > 0:
    #   T = self.model.config.sm_n_timesteps
    # else:
    #   T = 1000

    if batch_stats:
      variables={'params': params, 'batch_stats': batch_stats},
    else:
      variables={'params': params}

    conditioning = jnp.zeros((B,), dtype='uint8')
    # sample z_0 from the diffusion model
    rng, sample_rng = jax.random.split(rng)
    if hasattr(self.model, 'sample_zinit'):
      z_init = jnp.zeros((t_eval, B, H, W, C))
      z_init_0 = self.state.apply_fn(
        variables=variables,
        rng=sample_rng,
        batch=B,
        method=self.model.sample_zinit
      )
      z_init = z_init.at[0].set(z_init_0)
    elif hasattr(self.model, 'sample_eps'):
      z_init = self.state.apply_fn(
        variables=variables,
        rng=sample_rng,
        shape=(t_eval, B, 32, 17, C),
        axes=[2,3], 
        use_var1=True,
        method=self.model.sample_eps
      )
    else:
      z_init = jax.random.normal(sample_rng, (t_eval, B, H, W, C))
    
    # state = {
    #   "z_s": z_init,
    # }

    state = {
      "z_s": z_init,
      # "z_t": z_init,
      # "g_s": jnp.ones((t_eval, B)),
      # "g_t": jnp.ones((t_eval, B)),
      # "recon": z_init,
      # # "eps_hat": z_init,
      # "eps": z_init,
      # "eps_hat": z_init,
      # "base": dummy_inputs
    }

    def seq_body_fn(i, state):
      results = self.state.apply_fn(
          variables=variables,
          i=i,
          T=T_,
          z_t=state["z_s"][i],
          # dummy=state["base"],
          conditioning=conditioning,
          rng=rng,
          method=self.model.sample,
      )
      for k, v in results.items():
        state[k] = state[k].at[i+1].set(v)
      
      return state

    def single_body_fn(i, state):
      results = self.state.apply_fn(
          variables=variables,
          i=i,
          T=T_,
          z_t=state["z_s"][0],
          conditioning=conditioning,
          rng=rng,
          method=self.model.sample,
      )
      for k, v in results.items():
        state[k] = state[k].at[0].set(v)

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
          variables=variables,
          z_0=v,
          method=self.model.generate_x,
      )
      state[k] = samples.reshape(t_eval, B, H, W, C).transpose(1, 0, 2, 3, 4)
      # if k in ["z_s", "eps", "eps_hat", "recon", "z_t"]:
      #   v = jnp.reshape(v, (-1, H, W, C))
      #   samples = v
      #   # samples = self.state.apply_fn(
      #   #     variables={'params': params},
      #   #     z_0=v,
      #   #     method=self.model.generate_x,
      #   # )
      #   state[k] = samples.reshape(t_eval, B, H, W, C).transpose(1, 0, 2, 3, 4)
      # elif k in ['base']:
      #   state[k] = v
      # else:
      #   state[k] = v.transpose(1,0)

    return state

