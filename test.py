from dataset import create_eval_dataset, _preprocess_cifar10
import jax
import jax.numpy as jnp
from torchvision.utils import save_image
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

seed = 0
rng = jax.random.PRNGKey(seed)
sample_rng, denoise_rng, rng = jax.random.split(rng, 3)


B = 128
_ , eval_ds = create_eval_dataset(
    'cifar10',
    B,
    'test',
    sample_rng,
    _preprocess_cifar10,
)

eval_iter = iter(eval_ds)

def pixel_pdf(trg, mu, sigma):
    normed = (-0.5 * (trg[:,None,:] - mu[None,...]) ** 2 / sigma ** 2).sum(axis=-1, keepdims=True)
    # normed -= normed.max()
    normed /= (32 * 32 * 3)
    pdf = jnp.exp(normed) * jnp.sqrt(1 / (2 * sigma ** 2))
    return pdf

def four_pdf(trg, mu, sigma):
    mu = jnp.fft.rfft2(mu.reshape(-1, 32, 32, 3), axes=[1,2]).reshape(-1, 32*17*3)
    trg = jnp.fft.rfft2(trg.reshape(-1, 32, 32, 3), axes=[1,2]).reshape(-1, 32*17*3)
    sigma *= 32
    normed = (-0.5 * jnp.absolute(trg[:,None,:] - mu[None,...]) ** 2 / sigma ** 2).sum(axis=-1, keepdims=True)
    # normed -= normed.max()
    normed /= (32 * 32 * 3)
    pdf = jnp.exp(normed) * jnp.sqrt(1 / (2 * sigma ** 2))
    return pdf

def optimal_deonise(trg, imgs, alpha, sigma):
    # pdf = pixel_pdf(trg, alpha * imgs, sigma)
    pdf = four_pdf(trg, alpha * imgs, sigma)
    denoised = (pdf * alpha * imgs[None,...]).sum(axis=1) / pdf.sum(axis=1)
    return denoised

def toImg(batch):
    imgs = jnp.array(batch['images']).astype(jnp.float32)
    imgs = (imgs - 127.5) / 127.5
    return imgs.reshape(B, -1)
    
dataset = []
for i in range(100):
    batch = next(eval_iter)
    pimgs = toImg(batch)
    dataset.append(pimgs)

dataset = jnp.concatenate(dataset)[:B]
step = 100
alpha = jnp.arange(1/step, 1-1/step, 1/step)
sigma = 1 - alpha


# alpha = jnp.ones(step)
# sigma = jnp.exp(jnp.arange(-4, 3 - 1/step, (3 + 4)/step))

img_shape = (32 * 32 * 3,)
img_set = []
mag_set = []
for i in tqdm(range(step)):
    i_rng = jax.random.fold_in(denoise_rng, i)
    noise = jax.random.normal(i_rng, shape=(B,)+img_shape)
    pimgs = dataset[:B]
    # pimgs = toImg(batch)
    perturbed_img = pimgs * alpha[i] + noise * sigma[i]
    denoised_img = optimal_deonise(perturbed_img, dataset, alpha[i], sigma[i])
    img_set.append(jnp.concatenate([perturbed_img, denoised_img, pimgs]))
    eps_opt = (perturbed_img - denoised_img) / sigma[i]
    # eps_opt -= perturbed_img
    mag_set.append(eps_opt)


x_id = jnp.arange(32).reshape(-1, 1, 1).tile((1, 17, 1))
y_id = jnp.arange(17).reshape(1, -1, 1).tile((32, 1, 1))
idx = jnp.concatenate([x_id, y_id], axis=-1)
dist = jnp.square(idx).sum(axis=-1).reshape(-1)
dist_ord = jnp.argsort(dist)

for i in tqdm(range(step)):
    imgs = torch.tensor(np.array(img_set[i])).reshape(3 * B, 32, 32, 3)
    X, Y = 24, int(3*B / 24)
    imgs = imgs.reshape(X, Y, 32, 32, 3).permute(0,2,1,3,4).reshape(X*32, Y*32, 3)
    imgs = (imgs + 1) / 2
    save_image(imgs.permute(2,0,1), f'figure/cmp_{sigma[i]:.04f}.png')

    eps_opt = mag_set[i].reshape(-1, 32, 32, 3)
    eps_mag = jnp.absolute(jnp.fft.rfft2(eps_opt, axes=[1,2])).sum(axis=-1)
    eps_mag = eps_mag.reshape(B, -1)[dist_ord]
    eps_mag = torch.tensor(np.array(eps_mag))
    plt.clf()
    plt.figure(figsize=(20, 5))
    plt.plot(eps_mag.mean(axis=0), c='b', linewidth=0.3)
    plt.plot(eps_mag.mean(axis=0) - eps_mag.std(axis=0), c='r', linewidth=0.3)
    plt.plot(eps_mag.mean(axis=0) + eps_mag.std(axis=0), c='r', linewidth=0.3)
    plt.plot(eps_mag.max(axis=0)[0], c='g', linewidth=0.3)
    plt.plot(eps_mag.min(axis=0)[0], c='g', linewidth=0.3)
    plt.tight_layout()
    plt.savefig(f'figure/mag/cmp_{sigma[i]:.04f}.png')