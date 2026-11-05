# An Agentic Generative Framework for Joint Inverse Design of Material-Architecture of Mechanical Metamaterials

This repository contains the inference-time agentic pipeline accompanying the paper:

> **An Agentic Generative Framework for Joint Inverse Design of Material-Architecture of Mechanical Metamaterials**

It provides an LLM-orchestrated multi-agent system, built with [AutoGen](https://microsoft.github.io/autogen/), that wraps a trained shared-attribute VAE, a property predictor, and a conditional latent diffusion model to:

- **Forward simulation**: given a truss structure sentence and a base material (`E`, `nu`), predict the 21 independent components of the effective stiffness matrix `C` (GPa).
- **Inverse design**: given a target stiffness matrix (21 components), sample the conditional latent diffusion model and decode a candidate truss structure and base material that best achieves it.
- **Visualization**: render the 3D truss unit cell and a target-vs-predicted spider plot of the stiffness components.

A single orchestrating **Planning Agent** routes the request to a **Simulation Agent** or **Design Agent**, then to a matching **Plotting Agent**, and summarizes the result.

Notation follows the manuscript: **x = base material**, **transformer = structure**. A design is a coupled pair `(M_s, S)` of a base-material descriptor `M_s = (E_s, nu_s)` and a discrete structure sentence `S`.

This repository is **inference-only**: no training code is included.

## Stiffness components

The models work with the upper triangle of the 6x6 stiffness matrix, row by row, in the Voigt order of the data (1 = 11, 2 = 22, 3 = 33, **4 = 12**, 5 = 13, **6 = 23**):

```
C11 C12 C13 C14 C15 C16 C22 C23 C24 C25 C26 C33 C34 C35 C36 C44 C45 C46 C55 C56 C66
```


## Repository structure

```
.
├── agentic_ai.py           # Entry point: loads models, defines tools/agents, runs the team
├── vae_model.py             # Shared-attribute VAE, property predictor, tokenizer, greedy decoder, stiffness helpers
├── latent_diffusion.py      # Conditional latent diffusion sampler, denoiser, condition/latent transforms
├── Plotting_agent.py        # Unit-cell and spider-plot rendering
├── requirements.txt
├── .env                     # Fill in your OpenAI API key
├── data/
│   ├── char_dict.json               # Structure-sentence vocabulary (token -> id)
│   ├── org_dict.json                # Structure-sentence vocabulary (id -> token)
│   ├── structure_grammar.json       # Valid node labels and numbers of nodes per symmetry class
│   ├── scaler_x.pkl                 # Fitted scaler for the base-material inputs (E [GPa], nu)
│   ├── scaler_y.pkl                 # Fitted scaler for the 21 learned stiffness values
│   ├── condition_scaler.pkl         # Transform of the target stiffness into the diffusion condition
│   ├── latent_standardization.pkl   # Mean / std of the latent space used by the diffusion model
│   └── Materials_Properties.csv     # 4 reference materials, used only to colour the plots
├── checkpoints/
│   ├── vae.pt                       # Shared-attribute VAE weights
│   ├── property_predictor.pt        # Property predictor weights (21 heads)
│   └── denoiser.pth                 # Latent diffusion denoiser weights (21-component condition)
├── assets/                   # Example output images referenced below
└── outputs/                  # Generated unit-cell / spider-plot images (created at runtime)
```

## Requirements

- Python 3.10+
- An OpenAI API key (the agents use `gpt-4o` via `OpenAIChatCompletionClient`; set `AGENTIC_LLM` to use another model)

Install dependencies:

```bash
pip install -r requirements.txt
```


## Training and validation data

The full cleaned dataset is available [here](https://uofi.box.com/s/egy2ii5z7k4kibftl6vr5gudlgu3fuh5). It contains 1,093,180 structure–base-material–stiffness samples, from which the training and validation splits for the VAE, property predictor, and diffusion model were drawn.