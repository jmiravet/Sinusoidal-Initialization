# Sinusoidal Initialization: Time for a New Start

Official PyTorch implementation of **Sinusoidal Initialization**, as introduced in:

> **Sinusoidal Initialization, Time for a New Start**  
> Alberto Fernández-Hernández, Jose I. Mestre, Manuel F. Dolz, José Duato, Enrique S. Quintana-Ortí  
> NeurIPS 2025  
> [[OpenReview Page]](https://openreview.net/forum?id=FGliQVcrDZ) • [[PDF]](https://openreview.net/pdf?id=FGliQVcrDZ)

---

## 📝 Abstract

Initialization plays a critical role in Deep Neural Network training, directly influencing convergence, stability, and generalization. Common approaches such as Glorot and He initializations rely on randomness, which can produce uneven weight distributions across layer connections. In this paper, we introduce the Sinusoidal initialization, a novel deterministic method that employs sinusoidal functions to construct structured weight matrices expressly to improve the spread and balance of weights throughout the network while simultaneously fostering a more uniform, well‑conditioned distribution of neuron activation states from the very first forward pass. Because Sinusoidal initialization begins with weights and activations that are already evenly and efficiently utilized, it delivers consistently faster convergence, greater training stability, and higher final accuracy across a wide range of models, including convolutional neural networks, vision transformers, and large language models. On average, our experiments show an increase of 4.8 % in final validation accuracy and 20.9 % in convergence speed. By replacing randomness with structure, this initialization provides a stronger and more reliable foundation for Deep Learning systems.

---

## 🧠 What is Sinusoidal Initialization?

Sinusoidal Initialization replaces random weight sampling with **deterministic sinusoidal patterns**.  
Each neuron (or output channel) receives a unique wave defined by frequency and phase, producing:

- Smooth, structured, and diverse filters  
- Balanced, variance-normalized weight distributions  
- Uniform activation states from the first forward pass  

This yields **better-conditioned networks** that learn faster and more robustly.

Supported layers:

- `nn.Linear`
- `nn.Conv1d`, `nn.Conv2d`, `nn.Conv3d`
- `nn.MultiheadAttention`

---

## 🚀 How to use

Import `sinusoidal_init` and apply it to your PyTorch model:

```python
from initialize import sinusoidal_init

model.apply(sinusoidal_init)
```

---

## 📄 Cite This Work

If you use Sinusoidal Initialization in your research, please cite:

```bibtex
@inproceedings{fernandez-hernandez2025sinusoidal,
      author = {Fern\'{a}ndez-Hern\'{a}ndez, Alberto and Mestre, Jose and Dolz, Manuel F. and Duato, Jos\'{e} and Quintana-Orti, Enrique},
      booktitle = {Advances in Neural Information Processing Systems},
      doi = {10.52202/085713-2285},
      editor = {D. Belgrave and C. Zhang and H. Lin and R. Pascanu and P. Koniusz and M. Ghassemi and N. Chen},
      pages = {68045--68073},
      publisher = {Curran Associates, Inc.},
      title = {Sinusoidal Initialization, Time for a New Start},
      url = {https://proceedings.neurips.cc/paper_files/paper/2025/file/621dc667e65dce3ba19d882c7f2df1a7-Paper-Conference.pdf},
      volume = {38, Main Conference},
      year = {2025}
}
