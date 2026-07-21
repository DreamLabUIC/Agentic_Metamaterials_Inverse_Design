# From Target to Structure: Agent-Driven Generative Inverse Design of Multimaterial Truss Metamaterials

This repository contains the inference-time agentic pipeline accompanying the paper:

> **From Target to Structure: Agent-Driven Generative Inverse Design of Multimaterial Truss Metamaterials**

It provides an LLM-orchestrated multi-agent system, built with [AutoGen](https://microsoft.github.io/autogen/), that wraps a trained shared-attribute VAE, a property predictor, and a conditional latent diffusion model to:

- **Forward simulation** — given a truss structure sentence and a base material (`E`, `nu`), predict the twelve effective directional elastic properties.
- **Inverse design** — given twelve target elastic properties, sample the conditional latent diffusion model and decode a candidate truss structure and base material that best achieves them.
- **Visualization** — render the 3D truss unit cell and a target-vs-predicted radar chart of the elastic properties.

A single orchestrating **Planning Agent** routes the request to a **Simulation Agent** or **Design Agent**, then to a matching **Plotting Agent**, and summarizes the result.

Notation follows the manuscript: **x = base material**, **transformer = structure**. A design is a coupled pair `(M_s, S)` of a base-material descriptor `M_s = (E_s, nu_s)` and a discrete structure sentence `S`.

This repository is **inference-only** — no training code is included.

## Repository structure

```
.
├── agentic_ai.py           # Entry point: loads models, defines tools/agents, runs the team
├── vae_model.py             # Shared-attribute VAE, property predictor, tokenizer, greedy decoder
├── latent_diffusion.py      # Conditional latent diffusion sampler and denoiser
├── Plotting_agent.py        # Unit-cell and radar-chart rendering
├── requirements.txt
├── .env.example              # Copy to .env and fill in your OpenAI API key
├── data/
│   ├── char_dict.json               # Structure-sentence vocabulary (token -> id)
│   ├── org_dict.json                # Structure-sentence vocabulary (id -> token)
│   ├── scaler_x.pkl                 # Fitted scaler for the base-material inputs (E, nu)
│   ├── scaler_y.pkl                 # Fitted scaler for the 12 elastic properties
│   └── Materials_Properties.csv     # Reference materials used for unit-cell coloring_only few samples
├── checkpoints/
│   ├── vae.pt                       # Shared-attribute VAE weights
│   ├── property_predictor.pt        # Property predictor weights
│   └── denoiser.pth                 # Latent diffusion denoiser weights
├── assets/                   # Example output images referenced below
└── outputs/                  # Generated unit-cell / radar images (created at runtime)
```

## Requirements

- Python 3.10+
- An OpenAI API key (the agents use `gpt-4o` via `OpenAIChatCompletionClient`)

Install dependencies:

```bash
pip install -r requirements.txt
```

Libraries used:

| Library | Purpose |
|---|---|
| `torch` | Model definitions, tensor ops, checkpoint loading |
| `numpy` | Numerical arrays, beta schedules |
| `scipy` | `cKDTree` for lattice connectivity in unit-cell rendering |
| `pandas` | Reading the reference-materials CSV |
| `scikit-learn` | Unpickling the fitted `scaler_x.pkl` / `scaler_y.pkl` scalers |
| `matplotlib` | Unit-cell and radar-chart rendering |
| `python-dotenv` | Loading `OPENAI_API_KEY` from `.env` |
| `tqdm` | Optional progress bar during diffusion sampling |
| `autogen-agentchat`, `autogen-ext`, `autogen-core` | Multi-agent orchestration (Microsoft AutoGen) |

