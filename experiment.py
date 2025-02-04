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

from abc import ABC, abstractmethod
from collections import defaultdict
import functools
import os
from typing import Any, Tuple

from absl import logging
import chex
from clu import periodic_actions
from clu import parameter_overview
from clu import metric_writers
from clu import checkpoint
from flax.core.frozen_dict import unfreeze, FrozenDict
import flax.jax_utils as flax_utils
import flax
from jax._src.random import PRNGKey
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from optax._src import base
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
from flax.core.frozen_dict import FrozenDict

import vdm.train_state
import vdm.utils as utils
import vdm.dataset as dataset

class Experiment(ABC):
  """Boilerplate for training and evaluating VDM models."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.config = config

    # Set seed before initializing model.
    seed = config.training.seed
    self.rng = utils.with_verbosity("ERROR", lambda: jax.random.PRNGKey(seed))

    # initialize dataset
    logging.warning('=== Initializing dataset ===')
    self.rng, data_rng = jax.random.split(self.rng)
    self.train_iter, self.eval_iter, self.phase_iter = dataset.create_dataset(config, data_rng)

    # initialize model
    logging.warning('=== Initializing model ===')
    self.rng, model_rng = jax.random.split(self.rng)
    self.model, params = self.get_model_and_params(model_rng, next(self.phase_iter))
    parameter_overview.log_parameter_overview(params)

    # initialize train state
    logging.info('=== Initializing train state ===')
    self.state = vdm.train_state.TrainState.create(
        apply_fn=self.model.apply,
        variables=params,
        optax_optimizer=self.get_optimizer)
    self.lr_schedule = self.get_lr_schedule()

    # Restore from checkpoint
    ckpt_restore_dir = self.config.get('ckpt_restore_dir', 'None')
    if ckpt_restore_dir != 'None':
      ckpt_restore = checkpoint.Checkpoint(ckpt_restore_dir)
      checkpoint_to_restore = ckpt_restore.get_latest_checkpoint_to_restore_from()
      assert checkpoint_to_restore
      state_restore_dict = ckpt_restore.restore_dict(checkpoint_to_restore)
      self.state = restore_partial(self.state, state_restore_dict)
      del state_restore_dict, ckpt_restore, checkpoint_to_restore

    # initialize train/eval step
    logging.info('=== Initializing train/eval step ===')
    self.rng, train_rng = jax.random.split(self.rng)
    self.p_train_step = functools.partial(self.train_step, train_rng)
    self.p_train_step = functools.partial(jax.lax.scan, self.p_train_step)
    self.p_train_step = jax.pmap(self.p_train_step, "batch")

    self.rng, eval_rng, sample_rng, seq_rng, mse_rng = jax.random.split(self.rng, 5)
    self.p_eval_step = functools.partial(self.eval_step, eval_rng)
    self.p_eval_step = jax.pmap(self.p_eval_step, "batch")

    self.p_mse_step = functools.partial(self.mse_step, mse_rng)
    self.p_mse_step = jax.pmap(self.p_mse_step, "batch")

    self.p_sample = functools.partial(
        self.sample_fn,
        dummy_inputs=next(self.eval_iter)["images"][0],
        rng=sample_rng,
    )
    self.p_sample = utils.dist(
        self.p_sample, accumulate='concat', axis_name='batch')

    self.p_sample_seq = functools.partial(
        self.sample_fn,
        dummy_inputs=next(self.eval_iter)["images"][0],
        rng=seq_rng,
        return_seq=True
    )
    self.p_sample_seq = utils.dist(
        self.p_sample_seq, accumulate='concat', axis_name='batch')


    if self.config.model.sm_n_timesteps >0:
      self.T_list = [self.config.model.sm_n_timesteps, -1]
    else:
      self.T_list = [10, 100, 250, 500, 1000, -1]

        
    # if self.config.training.optimal_compare:
    if False:
      self.rng, data_model_rng = jax.random.split(self.rng)
      self.data_model, data_params, database = self.get_data_model(self.eval_iter, data_model_rng)

      self.p_opt_step = functools.partial(self.opt_step, params=data_params, database=database)
      # self.p_opt_step = functools.partial(jax.lax.scan, self.p_opt_step)
      self.p_opt_step = jax.pmap(self.p_opt_step, "batch")

    logging.info('=== Done with Experiment.__init__ ===')

  def get_lr_schedule(self):
    learning_rate = self.config.optimizer.learning_rate
    config_train = self.config.training
    # Create learning rate schedule
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=learning_rate,
        transition_steps=config_train.num_steps_lr_warmup
    )

    if self.config.optimizer.lr_decay:
      decay_fn = optax.linear_schedule(
          init_value=learning_rate,
          end_value=0,
          transition_steps=config_train.num_steps_train - config_train.num_steps_lr_warmup,
      )
      schedule_fn = optax.join_schedules(
          schedules=[warmup_fn, decay_fn], boundaries=[
              config_train.num_steps_lr_warmup]
      )
    else:
      schedule_fn = warmup_fn

    return schedule_fn

  def get_optimizer(self, lr: float) -> base.GradientTransformation:
    """Get an optax optimizer. Can be overided. """
    config = self.config.optimizer

    def decay_mask_fn(params):
      flat_params = flax.traverse_util.flatten_dict(unfreeze(params))
      flat_mask = {
          path: (path[-1] != "bias" and path[-2:]
                 not in [("layer_norm", "scale"), ("final_layer_norm", "scale")])
          for path in flat_params
      }
      return FrozenDict(flax.traverse_util.unflatten_dict(flat_mask))

    if config.name == "adamw":
      optimizer = optax.adamw(
          learning_rate=lr,
          mask=decay_mask_fn,
          **config.args,
      )
      if hasattr(config, "gradient_clip_norm"):
        clip = optax.clip_by_global_norm(config.gradient_clip_norm)
        optimizer = optax.chain(clip, optimizer)
    else:
      raise Exception('Unknow optimizer.')

    return optimizer

  @abstractmethod
  def get_data_model(self):
    """Return the data model."""
    ...

  @abstractmethod
  def get_model_and_params(self, rng: PRNGKey):
    """Return the model and initialized parameters."""
    ...

  @abstractmethod
  def sample_fn(self, *, dummy_inputs, rng, T, params) -> chex.Array:
    """Generate a batch of samples in [0, 255]. """
    ...

  @abstractmethod
  def loss_fn(self, params, batch_stat, batch, rng, is_train, T_eval) -> Tuple[float, Any]:
    """Loss function and metrics."""
    ...

  @abstractmethod
  def mse_fn(self, params, batch, rng, is_train, T_eval) -> Tuple[float, Any]:
    """Loss function and metrics."""
    ...

  def train_and_evaluate(self, workdir: str):
    logging.warning('=== Experiment.train_and_evaluate() ===')
    logging.info('Workdir: '+workdir)

    #if jax.process_index() == 0:
    #  if not tf.io.gfile.exists(workdir):
    #    tf.io.gfile.mkdir(workdir)

    config = self.config.training
    logging.info('num_steps_train=%d', config.num_steps_train)

    # Get train state
    state = self.state

    # Set up checkpointing of the model and the input pipeline.
    checkpoint_dir = os.path.join(workdir, 'checkpoints')
    ckpt = checkpoint.MultihostCheckpoint(checkpoint_dir, max_to_keep=5)
    checkpoint_to_restore = ckpt.get_latest_checkpoint_to_restore_from()
    if checkpoint_to_restore:
      state = ckpt.restore_or_initialize(state)
    initial_step = int(state.step)

    # Distribute training.
    state = flax_utils.replicate(state)
    T_list = self.T_list

    # Create logger/writer
    writer = utils.create_custom_writer(workdir, jax.process_index())
    if initial_step == 0:
      writer.write_hparams(dict(self.config))

    hooks = []
    report_progress = periodic_actions.ReportProgress(
        num_train_steps=config.num_steps_train, writer=writer)
    if jax.process_index() == 0:
      hooks += [report_progress]
      if config.profile:
        hooks += [periodic_actions.Profile(num_profile_steps=5,
                                           logdir=workdir)]

    step = initial_step
    substeps = config.substeps

    with metric_writers.ensure_flushes(writer):
      logging.info('=== Start of training ===')
      # the step count starts from 1 to num_steps_train
      while step < config.num_steps_train:
        is_last_step = step + substeps >= config.num_steps_train
        # One training step
        with jax.profiler.StepTraceAnnotation('train', step_num=step):
          batch = next(self.train_iter)
          B = batch["images"].shape[0]
          batch.update({"T_eval": tf.ones((B, substeps, 1)) * T_list[-1]})
          batch = jax.tree_map(jnp.asarray, batch)
          state, _train_metrics = self.p_train_step(state, batch)

          # grads = flax_utils.unreplicate(_train_metrics["grads"])
          # flat_grads = flax.traverse_util.flatten_dict(unfreeze(grads))
          # nan_grads = []
          # for k, v in flat_grads.items():
          #   if jnp.isnan(v).any():
          #     nan_grads.append(k)
          
          # if len(nan_grads):
          #   print(nan_grads)

        # Quick indication that training is happening.
        logging.log_first_n(
            logging.WARNING, 'Finished training step %d.', 3, step)
        for h in hooks:
          h(step)

        new_step = int(state.step[0])
        assert new_step == step + substeps
        step = new_step

        if step % config.steps_per_logging == 0 or is_last_step:
          logging.info('=== Writing scalars ===')
          metrics = flax_utils.unreplicate(_train_metrics['scalars'])

          def avg_over_substeps(x):
            assert x.shape[0] == substeps
            return float(x.mean(axis=0))

          metrics = jax.tree_map(avg_over_substeps, metrics)
          writer.write_scalars(step, metrics)

        if step % config.steps_per_eval == 0 or is_last_step or step == 1000:
          logging.info('=== Running eval ===')
          with report_progress.timed('eval'):
            for t in T_list[:-1]:
              eval_metrics = []
              for eval_step in range(config.num_steps_eval):
                batch = self.eval_iter.next()
                batch.update({"T_eval": tf.ones((B, 1)) * t})
                batch = jax.tree_map(jnp.asarray, batch)
                metrics = self.p_eval_step(
                    state.ema_params, state.ema_batch_stats, batch, flax_utils.replicate(eval_step))
                eval_metrics.append(metrics[f'scalars'])

              # average over eval metrics
              eval_metrics = utils.get_metrics(eval_metrics)
              eval_metrics = jax.tree_map(jnp.mean, eval_metrics)
              eval_metrics = {k + f'/{t:05d}':v for (k, v) in eval_metrics.items()}
              writer.write_scalars(step, eval_metrics)

        if step % config.steps_per_img == 0 or is_last_step or step == 1000:
            # print out a batch of images
            metrics = flax_utils.unreplicate(metrics)
            images = metrics['images']
            for t in T_list[:-1]:
              logging.info(f'=== Sampling Images (t_eval: {t:05d}) ===')
              samples = self.p_sample(params=state.ema_params, batch_stats=state.ema_batch_stats, T=flax_utils.replicate(jnp.ones([t,0])))
              samples = jnp.squeeze(samples['z_s'], axis=1)
              samples = utils.generate_image_grids(samples)[None, :, :, :]
              images[f'samples/{t:05d}'] = samples.astype(np.uint8)
            writer.write_images(step, images)

        if step % config.steps_per_save == 0 or is_last_step:
          with report_progress.timed('checkpoint'):
            ckpt.save(flax_utils.unreplicate(state))

  def evaluate(self, workdir):
    """Perform one evaluation."""
    logging.info('=== Experiment.evaluate() ===')

    checkpoint_dir = os.path.join(workdir, 'checkpoints-0')
    ckpt = checkpoint.Checkpoint(checkpoint_dir)
    state_dict = ckpt.restore_dict()
    params = flax.core.FrozenDict(state_dict['ema_params'])
    step = int(state_dict['step'])

    # Distribute training.
    params = flax_utils.replicate(params)
    T_list = self.T_list

    eval_logdir = os.path.join(workdir, 'eval')
    tf.io.gfile.makedirs(eval_logdir)
    writer = metric_writers.create_default_writer(
        eval_logdir, just_logging=jax.process_index() > 0)

    # from PIL import Image

    # def toImage(arr, path):
    #   igrid = utils.generate_image_grids(np.array(jnp.clip(arr * 127.5 + 127.5, a_min=0, a_max=255)).astype(np.uint8))
    #   Image.fromarray(np.array(igrid)).save(path)

    # # T_list = [1000,-1]
    # for t in T_list[:-1]:
    #   model_output = self.p_sample_seq(params=params, batch_stats=None, T=flax_utils.replicate(jnp.ones([t,0])))
    #   for k, v in model_output.items():
    #     v_shape = v.shape
    #     model_output[k] = v.reshape(4, -1, *v_shape[1:])
      
    #   optimal_output = self.p_opt_step(model_output)
    #   optimal_output = flax_utils.unreplicate(optimal_output)

    #   for k, v in model_output.items():
    #     v_shape = v.shape
    #     model_output[k] = v.reshape(-1, *v_shape[2:])
      
    #   print(1)


    # for t in T_list[:-1]:
    #   eval_metrics = []
    #   for eval_step in range(self.config.training.num_steps_eval):
    #     batch = self.eval_iter.next()
    #     batch.update({"T_eval": tf.ones((B, 1)) * t})
    #     batch = jax.tree_map(jnp.asarray, batch)
    #     metrics = self.p_eval_step(
    #         params, batch, flax_utils.replicate(eval_step))
    #     eval_metrics.append(metrics['scalars'])

    #   # average over eval metrics
    #   eval_metrics = utils.get_metrics(eval_metrics)
    #   eval_metrics = jax.tree_map(jnp.mean, eval_metrics)
    #   eval_metrics = {k + f'/{t:05d}':v for (k, v) in eval_metrics.items()}
    #   writer.write_scalars(step, eval_metrics)

    # sample a batch of images

    # save_images = True
    # if save_images:
    #   images = {}
    #   # # T_list = [10, 100, 250, 500, 1000]
    #   for t in T_list[:-1]:
    #     seq_samples = self.p_sample_seq(params=params, T=flax_utils.replicate(jnp.ones([t,0])))
    #     seq_samples = seq_samples["z_s"][:,-1,...]
    #     seq_samples = utils.generate_image_grids(seq_samples)[None, :, :, :]
    #     # samples = {'samples': samples.astype(np.uint8)}
    #     from PIL import Image
    #     images[f'eval_samples/{t:05d}'] = seq_samples.astype(np.uint8)

    #   for k,v in images.items():
    #     Image.fromarray(np.array(v[0])).save(f'vdm/evolve/cifar10_aa/{k}.png')

      # writer.write_images(step, images)

    cm = ['b','r','g','y','c']
    handles = []
    for tl_i, tl in enumerate(T_list[:-1][::-1]):
      handles.append(mpatches.Patch(color=cm[tl_i], label=f"{tl:05d}"))

    # Plot MSE-Timestep Graph
    plot_mse = True
    if plot_mse:
      plt.clf()
      plt.figure("snr_four", figsize=(20, 20))
      plt.figure("four", figsize=(20, 20))
      plt.figure("mse", figsize=(20, 20))      


      for ti, t in enumerate(T_list[:-1][-1:]):
        metric_list = defaultdict(list)
        for eval_step in range(self.config.training.num_steps_eval):
            batch = self.eval_iter.next()
            B = batch["images"].shape[0]
            batch.update({"T_eval": tf.ones((B, 1)) * t})
            batch = jax.tree_map(jnp.asarray, batch)
            metrics = self.p_mse_step(
                params, batch, flax_utils.replicate(eval_step))
            for k, v in metrics.items():
              metric_list[k].append(v.reshape(-1, *v.shape[2:]))
        
        for k, v in metric_list.items():
          metric_list[k] = jnp.vstack(v)

        from PIL import Image
        for i in range(1000):
          t = metric_list["timesteps"].reshape(-1)[i]
          f = metric_list["f"][i] * 128 + 128
          z = metric_list["z_t"][i] * 128 + 128
          r = metric_list["recon"][i] * 128 + 128
          eps = metric_list["eps"][i] * 128 + 128
          eps_hat = metric_list["eps_hat"][i] * 128 + 128

          output = np.vstack([f,z,r,eps,eps_hat,])
          Image.fromarray(np.array(output).astype(np.uint8)).save(f'vdm/samples/cifar10/{i}_{t}.png')

        if False:
          x = metric_list["timesteps"].reshape(-1)
          mse = np.sqrt(metric_list["loss_diff"].reshape(-1, 32, 32, 3))
          mse_snr_deriv = np.sqrt(metric_list["loss_diff"].reshape(-1, 32, 32, 3)) * metric_list["g_t_grad"].reshape(-1, 1, 1, 1)
          mse_snr = np.sqrt(metric_list["loss_diff"]).reshape(-1, 32, 32, 3) * metric_list["exp_g_t"].reshape(-1, 1, 1, 1)
        else:
          x = metric_list["timesteps"].reshape(-1)
          mse = np.sqrt(metric_list["loss_diff"].reshape(-1, 32, 17, 3))
          mse_snr_deriv_four = np.sqrt(metric_list["loss_diff"].reshape(-1, 32, 17, 3)) * metric_list["g_t_grad"]
          mse_snr_four = np.sqrt(metric_list["loss_diff"]).reshape(-1, 32, 17, 3) * metric_list["exp_g_t"]

        if False:
          plt.figure("mse")
          y = metric_list["exp_g_t"]
          # xy = np.vstack([x,y])
          # z = gaussian_kde(xy)(xy)
          plt.subplot(2,2,1)
          ax1 = plt.gca()
          ax2 = ax1.twinx()
          ax1.scatter(x,y,s=2,c='r',label="exp_g(t)") #, c=cm[ti], label=f"{t:05d}")
          ax1.set_ylabel("exp_g(t)")
          y = metric_list["g_t_grad"].reshape(-1)
          ax2.scatter(x,y, s=2,c='b',label="g(t)`")
          ax2.set_ylabel("g(t)`")

          plt.title('gamma(t) graph')
          # plt.legend(handles=handles)
          
          plt.subplot(2,2,2)
          y = jnp.mean(mse, axis=[1,2,3])
          plt.title('t-mse graph')
          # plt.legend(handles=handles)
          for ts in np.unique(x):
            ts_x = x[x == ts]
            ts_y = y[x == ts]
            # xy = np.vstack([ts_x,ts_y])
            if len(ts_y) > 1:
              z = gaussian_kde(ts_y)(ts_y)
              z = z / z.max()
            else:
              z = np.ones(len(ts_y))
            # plt.scatter(ts_x,ts_y, alpha=z, c=cm[ti], s=2, label=f"{t:05d}")
            plt.scatter(1-ts_x,ts_y, alpha=0.6, c=z, s=2)#,label=f"{t:05d}")
            # plt.legend(handles=handles)

          y = jnp.mean(mse_snr_deriv, axis=[1,2,3])
          plt.subplot(2,2,3)
          plt.title('t-mse_snr` graph')
          for ts in np.unique(x):
            ts_x = x[x == ts]
            ts_y = y[x == ts]
            # xy = np.vstack([ts_x,ts_y])
            if len(ts_y) > 1:
              z = gaussian_kde(ts_y)(ts_y)
              z = z / z.max()
            else:
              z = np.ones(len(ts_y))
            # plt.scatter(ts_x,ts_y, alpha=z, c=cm[ti], s=2, label=f"{t:05d}")
            plt.scatter(1 - ts_x,ts_y, alpha=0.6, c=z, s=2)#,label=f"{t:05d}")
            # plt.legend(handles=handles)

          y = jnp.mean(mse_snr, axis=[1,2,3])
          plt.subplot(2,2,4)
          plt.title('t-mse_snr graph')
          for ts in np.unique(x):
            ts_x = x[x == ts]
            ts_y = y[x == ts]
            # xy = np.vstack([ts_x,ts_y])
            if len(ts_y) > 1:
              z = gaussian_kde(ts_y)(ts_y)
              z = z / z.max()
            else:
              z = np.ones(len(ts_y))
            # plt.scatter(ts_x,ts_y, alpha=z, c=cm[ti], s=2, label=f"{t:05d}")
            plt.scatter(1 - ts_x,ts_y, alpha=0.6, c=z, s=2)#,label=f"{t:05d}")
            # plt.legend(handles=handles)
        
        # plt.figure("snr_four")
        # plt.title('t-mse_fourier graph')
        # # y = jnp.fft.fft2(mse_snr, axes=[1,2])
        # y = mse_snr_four
        # y = jnp.linalg.norm(jnp.fft.fftshift(jnp.absolute(y), axes=[1,2])[:,:,16,:], axis=-1)
        # for h_id in range(16):
        # 	plt.subplot(4, 4, h_id+1)
        # 	for ts in np.unique(x):
        # 		ts_x = x[x == ts]
        # 		ts_y = y[x == ts, 16 + h_id]
        # 		if len(ts_y) > 1:
        # 			z = gaussian_kde(ts_y)(ts_y)
        # 			# z = z / z.max()
        # 		else:
        # 			z = np.ones(len(ts_y))
        # 		# plt.scatter(ts_x, ts_y, alpha=z, c=cm[ti], s=2, label=f'{t:05d}')
        # 		plt.scatter(1 - ts_x, ts_y, alpha=0.6, c=z, s=2)#, label=f'{t:05d}')
        # 	plt.title(h_id)
        # 	# plt.legend(handles=handles)

        plt.figure("four")
        plt.title('t-mse_fourier graph')
        # y = jnp.fft.fft2(mse_snr_deriv, axes=[1,2])
        y = mse_snr_deriv_four
        y = jnp.linalg.norm(jnp.fft.fftshift(jnp.absolute(y), axes=[1,2])[:,:,16,:], axis=-1)
        alpha_t = metric_list['alpha_t'][:,:,16,0]
        data_y = jnp.linalg.norm(jnp.fft.fftshift(jnp.absolute(jnp.fft.fft2(metric_list["f"], axes=[1,2])), axes=[1,2])[:,:,16,:], axis=-1)
        for h_id in range(16):
          plt.subplot(4, 4, h_id+1)
          alpha_line = []
          for ts in np.unique(x):
            ts_x = x[x == ts]
            ts_y = y[x == ts, 16 + h_id]
            ts_alpha = alpha_t[x == ts, 16 + h_id]
            if len(ts_y) > 1:
              z = gaussian_kde(ts_y)(ts_y)
              # z = z / z.max()
            else:
              z = np.ones(len(ts_y))
            # plt.scatter(ts_x, ts_y, alpha=z, c=cm[ti], s=2, label=f'{t:05d}')
            plt.scatter(1 - ts_x, ts_y, alpha=0.6, c=z, s=2)#, label=f'{t:05d}')
            plt.scatter(1 - ts_x, ts_alpha * 20, alpha=0.8, c='r', s=5)
          plt.title(h_id)
          # plt.legend(handles=handles)
      
        plt.figure("mse")
        plt.tight_layout()
        plt.savefig('vdm/mse_plot.png')
        plt.figure("four")
        plt.tight_layout()
        plt.savefig('vdm/four_plot.png')
        plt.figure("snr_four")
        plt.tight_layout()
        plt.savefig('vdm/snr_four_plot.png')

    # Log evolution in fourier domain 
    plot_evolve=True
    if plot_evolve:
      plt.clf()
      plt.figure("evolve",figsize=(20,20))      
      for ti, t in enumerate(T_list[:-1][::-1]):
        fft_mag_list = []
        for eval_step in range(self.config.training.num_steps_eval // 10):
          seq_samples = self.p_sample_seq(params=params, T=flax_utils.replicate(jnp.ones([t,0])))
          from PIL import Image
          for k, v in seq_samples.items():
            img = utils.generate_image_grids(v[0])
            Image.fromarray(np.asarray(img).astype(np.uint8)).save(f'vdm/evolve/cifar10_conv/{k}.png')
          seq_samples = seq_samples['z_s']
          B, T, H, W, C = seq_samples.shape
          # seq_samples = seq_samples.reshape(t, -1, H, W, C).transpose(1,0,2,3,4)
          # B = B // t
          seq_samples = (seq_samples.astype(float) / 127.5) - 1
          fft_mag = jnp.linalg.norm(jnp.fft.fftshift(jnp.absolute(jnp.fft.fft2(seq_samples, axes=[2,3])), axes=[2,3])[:,:,:,16,:], axis=-1)
          fft_mag_list.append(np.array(fft_mag))

        fft_mag = np.vstack(fft_mag_list)
        for h_id in range(16):
          plt.subplot(4, 4, h_id+1)
          y_seq = []
          for tx in range(t):
            x = np.ones(len(fft_mag)) * (tx + 1) / t
            y = fft_mag[:,tx,16+h_id]
            if len(y) > 1:
              z = gaussian_kde(y)(y)
              z = z / z.max()
            else:
              z = np.ones(len(y))
            plt.scatter(x,y, alpha=z, c=cm[ti], s=1, label=f"{t:05d}")
            y_seq.append(y.mean())
          y_seq = np.array(y_seq)
          plt.plot((np.arange(t) + 1)/t, y_seq, c=cm[ti], label=f"{t:05d}")
          plt.legend(handles=handles)
          plt.title(h_id)

        plt.tight_layout()
        plt.savefig("vdm/evolve.png")

  def train_step(self, base_rng, state, batch):
    rng = jax.random.fold_in(base_rng, jax.lax.axis_index('batch'))
    rng = jax.random.fold_in(rng, state.step)

    grad_fn = jax.value_and_grad(self.loss_fn, has_aux=True)
    (_, metrics), grads = grad_fn(state.params, state.batch_stats, batch, rng=rng, is_train=True)
    grads = jax.lax.pmean(grads, "batch")
    stat_update = None
    if ("stat_update" in metrics) and (metrics["stat_update"] is not None):
      # stat_update = metrics.pop('stat_update')
      # stat_update = jax.lax.pmean(metrics.pop("stat_update"), "batch")
      stat_update = jax.lax.pmean(metrics["stat_update"]['batch_stats'], "batch")
      metrics.pop('stat_update')

    learning_rate = self.lr_schedule(state.step)
    new_state = state.apply_gradients(
        grads=grads, lr=learning_rate, ema_rate=self.config.optimizer.ema_rate, stat_update=stat_update)

    metrics['scalars'] = jax.tree_map(
        lambda x: jax.lax.pmean(x, axis_name="batch"), metrics['scalars'])
    metrics['scalars'] = {"train_" +
                          k: v for (k, v) in metrics['scalars'].items()}

    metrics['images'] = jax.tree_map(
        lambda x: utils.generate_image_grids(x)[None, :, :, :],
        metrics['images'])

    # metrics["grads"] = grads

    return new_state, metrics

  def eval_step(self, base_rng, params, batch_stat, batch, eval_step=0):
    rng = jax.random.fold_in(base_rng, jax.lax.axis_index('batch'))
    rng = jax.random.fold_in(rng, eval_step)


    _, metrics = self.loss_fn(params, batch_stat, batch, rng=rng, is_train=False)

    # summarize metrics
    metrics['scalars'] = jax.lax.pmean(
        metrics['scalars'], axis_name="batch")
    metrics['scalars'] = {
        "eval_" + k: v for (k, v) in metrics['scalars'].items()}

    metrics['images'] = jax.tree_map(
        lambda x: utils.generate_image_grids(x)[None, :, :, :],
        metrics['images'])
    
    metrics.pop('stat_update')

    return metrics
  
  def mse_step(self, base_rng, params, batch, eval_step=0):
    rng = jax.random.fold_in(base_rng, jax.lax.axis_index('batch'))
    rng = jax.random.fold_in(rng, eval_step)

    logs = self.mse_fn(params, batch, rng=rng, is_train=False)

    return logs

  def opt_step(self, batch, params, database):
    B, T, H, W, C = batch["eps"].shape
    state = {
      "g_s": batch["g_s"],
      "g_t": batch["g_t"],
      "eps": batch["eps"],
      "z_t": batch["z_t"],
      "z_s_opt": batch["z_s"].transpose(1,0,2,3,4),
      "eps_hat_opt": batch["z_s"].transpose(1,0,2,3,4),
      "opt_dn": batch["z_s"].transpose(1,0,2,3,4),
    }

    def body_fn(i, state):
      results = self.state.apply_fn(
          variables={'params': params},
          database=database,
          g_s=state["g_s"][:,i+1,...],
          g_t=state["g_t"][:,i+1,...],
          z_t=state["z_t"][:,i+1,...],
          eps=state["eps"][:,i+1,...],
          method=self.data_model.sample,
      )
      for k, v in results.items():
        if k in state:
          state[k] = state[k].at[i+1].set(v)
      
      return state

    state = jax.lax.fori_loop(
        lower=0, upper=T, body_fun=body_fn, init_val=state)
    
    result = {
      "z_s_opt": state["z_s_opt"],
      "eps_hat_opt": state["eps_hat_opt"],
      "opt_dn": state["opt_dn"]
    }

    for k, v in result.items():
      v = jnp.reshape(v, (-1, H, W, C))
      samples = v
      # samples = self.state.apply_fn(
      #     variables={'params': params},
      #     z_0=v,
      #     method=self.model.generate_x,
      # )
      result[k] = samples.reshape(T, B, H, W, C).transpose(1, 0, 2, 3, 4)

    return result

def copy_dict(dict1, dict2):
  if not isinstance(dict1, dict):
    assert not isinstance(dict2, dict)
    return dict2
  for key in dict1.keys():
    if key in dict2:
      dict1[key] = copy_dict(dict1[key], dict2[key])

  return dict1


def restore_partial(state, state_restore_dict):
  state_dict = flax.serialization.to_state_dict(state)
  state_dict = copy_dict(state_dict, state_restore_dict)
  state = flax.serialization.from_state_dict(state, state_dict)

  return state