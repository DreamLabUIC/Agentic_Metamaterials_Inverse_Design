"""
agentic_ai.py
=============

Agentic interface to the multimaterial truss design framework. A single
orchestrating agent, powered by a large language model and coordinated with
AutoGen, delegates to three specialized agents:

    * Simulation Agent : forward model. Given a structure sentence and a base
                         material (E_s, nu_s), predicts the twelve effective
                         directional properties via the shared-attribute VAE
                         and its property predictor.
    * Design Agent     : inverse model. Given a target property set, samples
                         the conditional latent diffusion model and decodes
                         candidate material-structure designs.
    * Plotting Agent   : renders the truss unit cell and the property radar
                         chart (predicted-only for a forward run, target-vs-
                         predicted for an inverse run).

This file loads the pre-trained models and exposes the tools; no training is
performed here.
"""

import asyncio
import json
import os
import pickle

import numpy as np
import torch
from torch.autograd import Variable

from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.tools import FunctionTool

from latent_diffusion import (
    LatentDenoiser, SubsampledDiffusionSampler, build_beta_schedule, select_sampling_timesteps,
)
from vae_model import (
    SharedAttributeVAE, PropertyPredictor, device,
    encode_structure, decode_structure, greedy_decode, word_tokenizer,
)
from Plotting_agent import (
    build_lattice, match_material, draw_unitcell_bezier_colored,
    radar_compare_groups2, plt,
)


# --------------------------------------------------------------------------
# 1. Environment
# --------------------------------------------------------------------------
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")


# --------------------------------------------------------------------------
# 2. Vocabulary, scalers, property names
# --------------------------------------------------------------------------
DATA_DIR = "data"
CHECKPOINT_DIR = "checkpoints"

with open(os.path.join(DATA_DIR, "char_dict.json")) as f:
    CHAR_DICT = json.load(f)
with open(os.path.join(DATA_DIR, "org_dict.json")) as f:
    ORG_DICT = {int(k): v for k, v in json.load(f).items()}

with open(os.path.join(DATA_DIR, "scaler_x.pkl"), "rb") as f:
    scaler_material = pickle.load(f)      # fitted on [E_s, nu_s]
with open(os.path.join(DATA_DIR, "scaler_y.pkl"), "rb") as f:
    scaler_props = pickle.load(f)         # fitted on the 12 properties

PROPERTY_NAMES = ["E1", "E2", "E3", "G12", "G13", "G23",
                  "nu12", "nu13", "nu21", "nu23", "nu31", "nu32"]


# --------------------------------------------------------------------------
# 3. Load pre-trained models
# --------------------------------------------------------------------------
vae = SharedAttributeVAE().to(device)
property_predictor = PropertyPredictor().to(device)
denoiser = LatentDenoiser(latent_dim=15, hidden_dim=512, num_layers=5,
                          cond_dim=12, cond_embed_dim=128).to(device)

vae.load_state_dict(
    torch.load(os.path.join(CHECKPOINT_DIR, "vae.pt"), map_location=device))
property_predictor.load_state_dict(
    torch.load(os.path.join(CHECKPOINT_DIR, "property_predictor.pt"), map_location=device))
denoiser.load_state_dict(
    torch.load(os.path.join(CHECKPOINT_DIR, "denoiser.pth"), map_location=device))

vae.eval()
property_predictor.eval()
denoiser.eval()
print("Models and scalers loaded.")


# --------------------------------------------------------------------------
# 4. Diffusion sampler
# --------------------------------------------------------------------------
DIFFUSION_STEPS = 100
N_CANDIDATES = 50
LATENT_DIM = 15

beta_schedule = build_beta_schedule("cosine", DIFFUSION_STEPS)
diffusion = SubsampledDiffusionSampler(
    kept_timesteps=select_sampling_timesteps(DIFFUSION_STEPS, "ddim100"),
    beta_schedule=beta_schedule,
    predicts_clean_latent=True,
    rescale_steps=True,
    clip_range=(-8.5, 8.5),
)


# --------------------------------------------------------------------------
# 5. Forward simulation tool
# --------------------------------------------------------------------------
def run_simulation(structure: str, E: float, nu: float) -> dict:
    """Predict the twelve effective properties of a truss design.

    Args:
        structure: structure sentence, e.g. "Cubic v0 e1 e3".
        E:  base-material Young's modulus.
        nu: base-material Poisson's ratio.

    Returns:
        dict mapping each property name to its predicted value, or
        {"error": <message>} on failure.
    """
    try:
        # reject any token outside the vocabulary
        unknown = [tok for tok in word_tokenizer(structure) if tok not in CHAR_DICT]
        if unknown:
            return {"error": f"Unknown token(s): {unknown}. "
                             f"Valid tokens: {list(CHAR_DICT.keys())}"}

        # encode the structure sentence and normalize the material input
        src = encode_structure(np.array([structure]), CHAR_DICT, max_len=127).to(device)
        material = torch.tensor(
            scaler_material.transform([[E, nu]]), dtype=torch.float32, device=device)

        pad_id = CHAR_DICT["_"]
        src = Variable(src).long()
        src_mask = (src != pad_id).unsqueeze(-2)

        # encode -> latent mean -> property prediction (no sampling noise)
        with torch.no_grad():
            latent, _, _, _ = vae.encoder(src, src_mask, material)
            pred_norm = property_predictor(latent)

        pred = scaler_props.inverse_transform(pred_norm.cpu().numpy())[0]
        return {name: float(round(v, 6)) for name, v in zip(PROPERTY_NAMES, pred)}
    except Exception as exc:
        return {"error": str(exc)}


simulation_tool = FunctionTool(
    run_simulation,
    description=(
        "Forward simulation: given a truss structure sentence (e.g. 'Cubic v0 "
        "e1 e3') and base-material properties E and nu, predict the 12 effective "
        "elastic properties (E1, E2, E3, G12, G13, G23, nu12, nu13, nu21, nu23, "
        "nu31, nu32)."
    ),
)


# --------------------------------------------------------------------------
# 6. Inverse design tool
# --------------------------------------------------------------------------
def run_inverse_design(
    E1: float, E2: float, E3: float,
    G12: float, G13: float, G23: float,
    nu12: float, nu21: float, nu13: float,
    nu31: float, nu23: float, nu32: float,
) -> dict:
    """Generate a truss design that meets the twelve target properties.

    Samples N candidate designs from the conditional diffusion model, decodes
    each into a structure sentence and base material, and returns the candidate
    whose predicted properties are closest to the target.

    Returns a dict with the recommended structure sentence, base material
    (E, nu), and the properties that candidate achieves; or {"error": ...}.
    """
    try:
        target_real = np.array([[E1, E2, E3, G12, G13, G23,
                                 nu12, nu21, nu13, nu31, nu23, nu32]],
                               dtype=np.float32)
        target = torch.tensor(
            scaler_props.transform(target_real), dtype=torch.float32, device=device)

        with torch.no_grad():
            # sample a batch of latents conditioned on the target
            cond = target.repeat(N_CANDIDATES, 1)
            latents = diffusion.run_reverse_chain(
                denoiser, (N_CANDIDATES, LATENT_DIM), condition=cond)[-1]
            latents = torch.as_tensor(latents, dtype=torch.float32, device=device)

            # decode each latent into a structure sentence and base material
            tokens, material = greedy_decode(vae, latents, src_mask=None, use_gpu=True)
            sentences = decode_structure(tokens, ORG_DICT)

            # rank candidates by squared error against the target
            pred = property_predictor(latents)
            errors = torch.sum((pred - target) ** 2, dim=1)
            best = int(torch.argmin(errors).item())

            best_props = scaler_props.inverse_transform(
                pred[best].unsqueeze(0).cpu().numpy())[0]
            best_material = scaler_material.inverse_transform(
                material[best].unsqueeze(0).cpu().numpy())[0]

        achieved = {name: float(round(v, 6))
                    for name, v in zip(PROPERTY_NAMES, best_props)}
        return {
            "structure": sentences[best],
            "E": float(round(best_material[0], 6)),
            "nu": float(round(best_material[1], 6)),
            "achieved_properties": achieved,
        }
    except Exception as exc:
        return {"error": str(exc)}


inverse_design_tool = FunctionTool(
    run_inverse_design,
    description=(
        "Inverse design: given 12 target elastic properties (E1, E2, E3, G12, "
        "G13, G23, nu12, nu13, nu21, nu23, nu31, nu32), use the latent diffusion "
        "model to propose the truss structure sentence and base material (E, nu) "
        "that best achieve them."
    ),
)


# --------------------------------------------------------------------------
# 7. Plotting tools
# --------------------------------------------------------------------------
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_unit_cell(structure: str, E: float, nu: float,
                   save_path: str = None) -> dict:
    """Render the 3D truss unit cell for a structure sentence and base material.

    Colors the cell by the reference material closest to (E, nu). Returns the
    saved image path and the matched material info, or {"error": ...}.
    """
    try:
        points, connectivity, connected, _ = build_lattice(structure)
        color, element, material_name = match_material([E, nu], verbose=False)

        ax = draw_unitcell_bezier_colored(
            points, connectivity, control_point_offset_ratio=0, R=5, color=color)

        if save_path is None:
            save_path = os.path.join(
                OUTPUT_DIR, "unitcell_%s.png" % structure.strip().replace(" ", "_"))
        ax.figure.savefig(save_path, dpi=600, bbox_inches="tight", transparent=True)
        plt.close(ax.figure)
        os.startfile(os.path.abspath(save_path))

        return {"image_path": save_path, "matched_material": material_name,
                "matched_element": element, "matched_color": color,
                "connected": bool(connected)}
    except Exception as exc:
        return {"error": str(exc)}


unit_cell_plot_tool = FunctionTool(
    plot_unit_cell,
    description=(
        "Render the 3D truss unit cell for a structure sentence (e.g. 'Cubic v0 "
        "v1 v2 v3') and base material (E, nu), colored by the closest reference "
        "material. Returns the image path and matched material info."
    ),
)


def plot_radar(predicted_properties: dict, target_properties: dict = None,
               E: float = None, nu: float = None,
               save_path: str = None) -> dict:
    """Draw a 12-property radar chart of the effective properties.

    Pass only ``predicted_properties`` for a forward-simulation result (single
    polygon). Pass both ``predicted_properties`` and ``target_properties`` for
    an inverse-design comparison (target vs. achieved). Returns the image path
    and the mode, or {"error": ...}.
    """
    try:
        pred_vec = np.array([predicted_properties[n] for n in PROPERTY_NAMES],
                            dtype=float)
        target_vec = None
        if target_properties is not None:
            target_vec = np.array([target_properties[n] for n in PROPERTY_NAMES],
                                  dtype=float)

        material_color = None
        if E is not None and nu is not None:
            material_color, _, _ = match_material([E, nu], verbose=False)

        ax = radar_compare_groups2(pred_vec, target_vec, matColor=material_color)
        mode = "compare" if target_vec is not None else "predicted_only"

        if save_path is None:
            save_path = os.path.join(OUTPUT_DIR, "radar_%s.png" % mode)
        ax.figure.savefig(save_path, dpi=600, bbox_inches="tight", transparent=True)
        plt.close(ax.figure)
        os.startfile(os.path.abspath(save_path))

        return {"image_path": save_path, "mode": mode}
    except Exception as exc:
        return {"error": str(exc)}


radar_plot_tool = FunctionTool(
    plot_radar,
    description=(
        "Draw a 12-property radar chart (E1, E2, E3, G12, G13, G23, nu12, nu13, "
        "nu21, nu23, nu31, nu32). Pass only predicted_properties for a forward "
        "result; pass both predicted_properties and target_properties for an "
        "inverse-design target-vs-achieved comparison."
    ),
)


# --------------------------------------------------------------------------
# 8. Agents
# --------------------------------------------------------------------------
model_client = OpenAIChatCompletionClient(model="gpt-4o", api_key=API_KEY)

simulation_agent = AssistantAgent(
    name="SimulationAgent",
    description=("Forward model: given a structure sentence and base material "
                 "(E, nu), predicts the 12 effective elastic properties."),
    model_client=model_client,
    tools=[simulation_tool],
    reflect_on_tool_use=True,
    system_message="""
You are the Simulation Agent for a truss metamaterial framework.

Tool: run_simulation(structure, E, nu) -> 12 predicted elastic properties
  E1, E2, E3   Young's moduli along the three axes
  G12, G13, G23  shear moduli
  nu12, nu21, nu13, nu31, nu23, nu32  Poisson's ratios

Steps:
1. Call run_simulation with the user inputs.
2. Present the result as a Property | Value table.
3. Give a short physical reading (anisotropy, stiffest direction, notable
   Poisson effects).
4. On error, explain it and tell the user how to fix the input.
""",
)

design_agent = AssistantAgent(
    name="DesignAgent",
    description=("Inverse model: given 12 target properties, proposes the truss "
                 "structure and base material that best achieve them."),
    model_client=model_client,
    tools=[inverse_design_tool],
    reflect_on_tool_use=True,
    system_message="""
You are the Design Agent for a truss metamaterial framework.

Tool: run_inverse_design(12 target properties) -> a candidate design
  returns: structure sentence, base material (E, nu), and achieved_properties.

Steps:
1. Call run_inverse_design with the target properties.
2. Present the recommended structure sentence and base material (E, nu).
3. Show a Property | Target | Achieved table for all 12 properties.
4. Note how closely the design meets the target and flag notable deviations.
5. On error, explain it and ask the user to check the inputs.
""",
)

sim_plotting_agent = AssistantAgent(
    name="SimPlottingAgent",
    description=("Visualizes a forward result: draws only the predicted-property "
                 "radar chart (no unit cell, no target)."),
    model_client=model_client,
    tools=[radar_plot_tool],
    reflect_on_tool_use=True,
    system_message="""
You visualize forward-simulation results.

Tool: plot_radar. Call it with predicted_properties set to the 12 properties
from SimulationAgent and pass E and nu for coloring. Never pass
target_properties -- a forward run has no target. Report the image path.
""",
)

design_plotting_agent = AssistantAgent(
    name="DesignPlottingAgent",
    description=("Visualizes an inverse result: draws the 3D unit cell and a "
                 "target-vs-achieved radar chart."),
    model_client=model_client,
    tools=[unit_cell_plot_tool, radar_plot_tool],
    reflect_on_tool_use=True,
    system_message="""
You visualize inverse-design results.

Tools: plot_unit_cell and plot_radar.

Steps:
1. Call plot_unit_cell with the structure sentence and base material (E, nu)
   from DesignAgent.
2. Call plot_radar with predicted_properties = achieved_properties,
   target_properties = the user's original targets, and E/nu for coloring.
3. Report both image paths and the matched material (element, name, color).
   Ensure transparent background at 600 DPI.
""",
)

planning_agent = AssistantAgent(
    name="PlanningAgent",
    description=("Orchestrator: routes the request to the Simulation or Design "
                 "agent, then to the matching plotting agent, then summarizes."),
    model_client=model_client,
    system_message="""
You are the orchestrating agent for a truss metamaterial framework.

Team:
  SimulationAgent     forward: structure + material -> 12 properties
  DesignAgent         inverse: 12 target properties -> structure + material
  SimPlottingAgent    radar chart for a forward result
  DesignPlottingAgent unit cell + target-vs-achieved radar for an inverse result

Rules:
  - If the user gives a structure + material and wants properties -> SimulationAgent.
  - If the user gives target properties and wants a design -> DesignAgent.
  - Delegate with:  1. <AgentName> : <task with all inputs>
  - After SimulationAgent, delegate to SimPlottingAgent with the structure,
    material (E, nu), and predicted properties.
  - After DesignAgent, delegate to DesignPlottingAgent with the structure,
    material (E, nu), achieved properties, and original targets.
  - You never call tools yourself.
  - After the plotting agent reports, summarize in plain language and end with
    "TERMINATE".
""",
)


# --------------------------------------------------------------------------
# 9. Team
# --------------------------------------------------------------------------
termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(8)

selector_prompt = """
Select the next agent to act.

{roles}

Current conversation:
{history}

From {participants}, pick the next agent. Rules:
- PlanningAgent goes first on a new task.
- After PlanningAgent delegates, pick the agent it named.
- After SimulationAgent, pick SimPlottingAgent.
- After DesignAgent, pick DesignPlottingAgent.
- After a plotting agent, return to PlanningAgent to summarize and terminate.
Select only one agent.
"""

team = SelectorGroupChat(
    participants=[planning_agent, simulation_agent, design_agent,
                  sim_plotting_agent, design_plotting_agent],
    model_client=model_client,
    termination_condition=termination,
    selector_prompt=selector_prompt,
    allow_repeated_speaker=True,
)


# --------------------------------------------------------------------------
# 10. Entry point
# --------------------------------------------------------------------------
async def main():
    from autogen_agentchat.ui import Console

    # Forward-task example:
    task = ("Run a simulation for the structure 'Orthotropic v3 e0 e2 e5 e10 f0' "
            "with E=210.0 and nu=0.3.")

    # Inverse-task example:
    #task = ("I need a truss design that achieves the following elastic "
    #         "properties: E1=0.9, E2=0.9, E3=0.9, G12=0.09, G13=0.09, "
    #         "G23=0.09, nu12=0, nu21=0, nu13=0, nu31=0, nu23=0, nu32=0.")

    await Console(team.run_stream(task=task))


if __name__ == "__main__":
    asyncio.run(main())
