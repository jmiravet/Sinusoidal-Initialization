# Sinusoidal Initialization: Time for a New Start

Official PyTorch implementation of **Sinusoidal Initialization**, as introduced in:

> **Sinusoidal Initialization, Time for a New Start**  
> Alberto Fernández-Hernández, Jose I. Mestre, Manuel F. Dolz, Jose Duato, Enrique S. Quintana-Ortí  
> arXiv:2505.12909, 2025  
> [[arXiv link]](https://arxiv.org/abs/2505.12909)

---

## 📝 Abstract

Initialization plays a critical role in Deep Neural Network training, directly influencing convergence, stability, and generalization. Common approaches such as Glorot and He initializations rely on randomness, which can produce uneven weight distributions across layer connections.

In this paper, we introduce the *Sinusoidal* initialization, a novel **deterministic** method that employs sinusoidal functions to construct structured weight matrices, expressly designed to improve the spread and balance of weights throughout the network while simultaneously fostering a more uniform, well-conditioned distribution of neuron activation states from the very first forward pass.

Because *Sinusoidal* initialization begins with weights and activations that are already evenly and efficiently utilized, it delivers:

- **Consistently faster convergence**
- **Greater training stability**
- **Higher final accuracy**

across a wide range of models, including convolutional neural networks, vision transformers, and large language models.  
On average, our experiments show:

- **+4.9%** improvement in final validation accuracy  
- **+20.9%** improvement in convergence speed  

By replacing randomness with structure, this initialization provides a stronger and more reliable foundation for Deep Learning systems.

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

---

## 📄 Cite This Work

If you use Sinusoidal Initialization in your research, please cite:

```bibtex
@misc{fernándezhernández2025sinusoidalinitializationtimenew,
      title={Sinusoidal Initialization, Time for a New Start}, 
      author={Alberto Fernández-Hernández and Jose I. Mestre and Manuel F. Dolz and Jose Duato and Enrique S. Quintana-Ortí},
      year={2025},
      eprint={2505.12909},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.12909}, 
}
