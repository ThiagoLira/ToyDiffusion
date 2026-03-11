# ToyDiffusion

A minimal diffusion model you can run on your laptop. The code implements the core ideas from the [DDPM paper](https://arxiv.org/abs/2006.11239) (Ho et al., 2020) on a toy 2D dataset (Homer Simpson pixel coordinates), so you can see the full forward/reverse diffusion pipeline without needing a GPU or a U-Net.

[Medium post](https://thiago-lira.medium.com/a-toy-diffusion-model-you-can-run-on-your-laptop-20e9e5a83462)

## Running

Run all cells in the `RunDiffusion.ipynb` notebook. The code and comments reference the relevant DDPM paper equations throughout.

## Project structure

- `diffusion.py` — Forward diffusion (`q_sample`), reverse denoising (`denoise_with_mu`), posterior computation (`posterior_q`), sinusoidal position encoding, and the `Denoising` MLP model.
- `RunDiffusion.ipynb` — End-to-end training and sampling, with inline explanations tied to the paper.
- `THEORY/` — Reading guides and reference materials for understanding diffusion models and flow matching.

![image](https://miro.medium.com/max/1400/1*Nv_K1Ul7VUwVwHeb7PIrQQ.gif)
