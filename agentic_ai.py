"""
agentic_ai.py
=============

Agentic interface to the multimaterial truss design framework. A single
orchestrating agent, powered by a large language model and coordinated with
AutoGen, delegates to three specialized agents:

    * Simulation Agent : forward model. Given a structure sentence and a base
                         material (E_s, nu_s), predicts the 21 independent
                         components of the effective stiffness matrix C (GPa)
                         via the shared-attribute VAE and its property predictor.
    * Design Agent     : inverse model. Given a target stiffness matrix (21
                         components), samples the conditional latent diffusion
                         model and decodes candidate material-structure designs.
    * Plotting Agent   : renders the truss unit cell and the stiffness spider
                         plot (predicted-only for a forward run, target-vs-
                         predicted for an inverse run).

Stiffness components: upper triangle of the 6x6 matrix, row by row, in the
Voigt order of the data (1 = 11, 2 = 22, 3 = 33, 4 = 12, 5 = 13, 6 = 23):
    C11 C12 C13 C14 C15 C16 C22 C23 C24 C25 C26 C33 C34 C35 C36 C44 C45 C46 C55 C56 C66
Entries that are zero for a symmetry class are set to exactly 0.

This file loads the pre-trained models and exposes the tools; no training is
performed here.
"""

import asyncio
import json
import os
import pickle
import subprocess
import sys
from typing import Optional

import numpy as np
import torch

from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.tools import FunctionTool

from latent_diffusion import (
    LatentDenoiser, SubsampledDiffusionSampler, build_beta_schedule, select_sampling_timesteps,
    StiffnessCondition, LatentStandardization,
)
from vae_model import (
    SharedAttributeVAE, PropertyPredictor, device, STIFFNESS_NAMES,
    encode_structure, greedy_decode, word_tokenizer, check_structure,
    learned_to_C, apply_symmetry_zeros, relative_C_error,
)
from Plotting_agent import (
    build_lattice, match_material, draw_unitcell_bezier_colored,
    radar_compare_groups_C, plt,
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
with open(os.path.join(DATA_DIR, "structure_grammar.json")) as f:
    GRAMMAR = json.load(f)               # node labels / numbers of nodes per symmetry class

with open(os.path.join(DATA_DIR, "scaler_x.pkl"), "rb") as f:
    scaler_material = pickle.load(f)      # fitted on [E_s (GPa), nu_s]
with open(os.path.join(DATA_DIR, "scaler_y.pkl"), "rb") as f:
    scaler_props = pickle.load(f)         # fitted on the 21 learned stiffness values

condition_scaler = StiffnessCondition(os.path.join(DATA_DIR, "condition_scaler.pkl"))
latent_standardization = LatentStandardization(
    os.path.join(DATA_DIR, "latent_standardization.pkl"), device=device)

PROPERTY_NAMES = STIFFNESS_NAMES          # C11 ... C66 (21)

# Base-material database. The full framework snaps every decoded (E_s, nu_s)
# to the nearest alloy of a reference database (Ansys Granta material data).
# That database is proprietary and is NOT distributed with this package:
# without it, the decoded material properties are used directly, and the
# 4 materials in Materials_Properties.csv are only used to colour the plots.
# If you have the database, place it at data/Properties_samples0_with_material.csv
# (columns YoungsModulus_Mpsi, PoissonsRatio, BaseMaterial) to enable snapping.
MATERIAL_DATABASE = os.path.join(DATA_DIR, "Properties_samples0_with_material.csv")
USE_MATERIAL_DATABASE = os.path.exists(MATERIAL_DATABASE)
REFERENCE_MATERIALS = (MATERIAL_DATABASE if USE_MATERIAL_DATABASE
                       else os.path.join(DATA_DIR, "Materials_Properties.csv"))

if USE_MATERIAL_DATABASE:
    import pandas as pd
    _db = pd.read_csv(MATERIAL_DATABASE)
    _db["E"] = np.floor(_db["YoungsModulus_Mpsi"] * 6.8947572932 + 0.5)     # Mpsi -> GPa, 1 GPa precision
    _alloys = _db.groupby(["E", "PoissonsRatio"])["BaseMaterial"].apply(
        lambda s: "; ".join(sorted(set(s))))
    ALLOYS_RAW = np.array(_alloys.index.tolist(), dtype=float)       # distinct (E_s, nu_s) pairs
    ALLOY_NAMES = _alloys.to_numpy()
    ALLOYS_SCALED = torch.tensor(scaler_material.transform(ALLOYS_RAW),
                                 dtype=torch.float32, device=device)
    print("Material database: %d alloys -> %d distinct (E, nu) pairs." % (len(_db), len(ALLOYS_RAW)))


# --------------------------------------------------------------------------
# 3. Load pre-trained models
# --------------------------------------------------------------------------
vae = SharedAttributeVAE().to(device)
property_predictor = PropertyPredictor().to(device)
denoiser = LatentDenoiser(latent_dim=15, hidden_dim=512, num_layers=5,
                          cond_dim=21, cond_embed_dim=128).to(device)

vae.load_state_dict(
    torch.load(os.path.join(CHECKPOINT_DIR, "vae.pt"), map_location=device, weights_only=True))
property_predictor.load_state_dict(
    torch.load(os.path.join(CHECKPOINT_DIR, "property_predictor.pt"), map_location=device, weights_only=True))
denoiser.load_state_dict(
    torch.load(os.path.join(CHECKPOINT_DIR, "denoiser.pth"), map_location=device, weights_only=True))

vae.eval()
property_predictor.eval()
denoiser.eval()
print("Models and scalers loaded.")


# --------------------------------------------------------------------------
# 4. Diffusion sampler
# --------------------------------------------------------------------------
DIFFUSION_STEPS = 100
N_CANDIDATES = 100
LATENT_DIM = 15

beta_schedule = build_beta_schedule("cosine", DIFFUSION_STEPS)
diffusion = SubsampledDiffusionSampler(
    kept_timesteps=select_sampling_timesteps(DIFFUSION_STEPS, "ddim100"),
    beta_schedule=beta_schedule,
    predicts_clean_latent=True,
    rescale_steps=True,
    clip_range=(-6.0, 6.0),               # in units of the standardised latent
)


# --------------------------------------------------------------------------
# 5. Shared helper: structure + material -> C (GPa)
# --------------------------------------------------------------------------
@torch.no_grad()
def predict_stiffness(tokens, material_scaled, Es, sentences):
    """Encode structure tokens + base material, take the encoder MEAN (not a
    sampled z), predict the 21 learned values and convert them to C in GPa
    with the zeros of the symmetry class."""
    pad_id = CHAR_DICT["_"]
    _, mu, _, _ = vae.encoder(tokens, (tokens != pad_id).unsqueeze(-2), material_scaled)
    learned = scaler_props.inverse_transform(property_predictor(mu).cpu().numpy())
    return apply_symmetry_zeros(learned_to_C(learned, Es), sentences)


def _as_dict(C21):
    return {name: float(f"{v:.6g}") + 0.0 for name, v in zip(PROPERTY_NAMES, C21)}   # + 0.0: no -0.0


# --------------------------------------------------------------------------
# 6. Forward simulation tool
# --------------------------------------------------------------------------
def run_simulation(structure: str, E: float, nu: float) -> dict:
    """Predict the 21 stiffness components of a truss design.

    Args:
        structure: structure sentence, e.g. "Cubic v0 e1 e3".
        E:  base-material Young's modulus (GPa).
        nu: base-material Poisson's ratio.

    Returns:
        dict mapping C11 ... C66 to the predicted value in GPa, or
        {"error": <message>} on failure.
    """
    try:
        structure = " ".join(word_tokenizer(structure))
        problem = check_structure(structure, GRAMMAR)
        if problem:
            return {"error": f"Invalid structure '{structure}': {problem}."}
        lo, hi = scaler_material.data_min_, scaler_material.data_max_
        if not (lo[0] <= E <= hi[0] and lo[1] <= nu <= hi[1]):
            return {"error": f"Base material outside the training range: E in [{lo[0]:g}, {hi[0]:g}] GPa, "
                             f"nu in [{lo[1]:g}, {hi[1]:g}]."}

        tokens = encode_structure(np.array([structure]), CHAR_DICT, max_len=127).to(device)
        material = torch.tensor(
            scaler_material.transform([[E, nu]]), dtype=torch.float32, device=device)
        C = predict_stiffness(tokens, material, [E], [structure])[0]
        return _as_dict(C)
    except Exception as exc:
        return {"error": str(exc)}


simulation_tool = FunctionTool(
    run_simulation,
    description=(
        "Forward simulation: given a truss structure sentence (e.g. 'Cubic v0 "
        "e1 e3') and base-material properties E (GPa) and nu, predict the 21 "
        "components of the effective stiffness matrix C11 ... C66 (GPa)."
    ),
)


# --------------------------------------------------------------------------
# 7. Inverse design tool
# --------------------------------------------------------------------------
@torch.no_grad()
def run_inverse_design(
    C11: float, C12: float, C13: float, C14: float, C15: float, C16: float,
    C22: float, C23: float, C24: float, C25: float, C26: float,
    C33: float, C34: float, C35: float, C36: float,
    C44: float, C45: float, C46: float,
    C55: float, C56: float,
    C66: float,
) -> dict:
    """Generate a truss design whose stiffness matrix matches the target.

    Samples N candidate latents from the conditional diffusion model and, for
    every candidate:
      1. decodes the latent into a structure sentence and a base material;
      2. keeps it only if the sentence is a valid structure;
      3. base material: snaps the decoded (E_s, nu_s) to the nearest alloy of
         the material database if it is available (see section 2); otherwise
         uses the decoded material properties directly (the database, Ansys
         Granta material data, is proprietary and not distributed);
      4. re-encodes the decoded structure together with that material and
         takes the encoder MEAN (the actual latent of that design);
      5. predicts its stiffness from that mean and compares it with the target.
    The candidate with the smallest ||C - C_target||_F / ||C_target||_F is
    returned.

    Returns a dict with the recommended structure sentence, base material
    (E, nu), achieved C (GPa) and the relative error; or {"error": ...}.
    """
    try:
        target = np.array([[C11, C12, C13, C14, C15, C16, C22, C23, C24, C25, C26,
                            C33, C34, C35, C36, C44, C45, C46, C55, C56, C66]], dtype=float)
        cond = torch.tensor(condition_scaler.transform(target), dtype=torch.float32,
                            device=device).repeat(N_CANDIDATES, 1)

        # 1. sample latents conditioned on the target, back to the VAE latent
        z_std = diffusion.run_reverse_chain(
            denoiser, (N_CANDIDATES, LATENT_DIM), condition=cond)[-1]
        latents = latent_standardization.to_latent(z_std)

        # 2. decode structure sentences and base materials
        tokens, sentences, ended, material = greedy_decode(vae, latents, CHAR_DICT, ORG_DICT)
        valid = np.array([bool(e) and check_structure(s, GRAMMAR) is None
                          for s, e in zip(sentences, ended)])
        if not valid.any():
            return {"error": "No valid structure among the generated candidates; try again."}
        vi = np.where(valid)[0]

        # 3. base material: nearest database alloy, or the decoded material (see section 2)
        alloy = None
        if USE_MATERIAL_DATABASE:
            alloy = torch.cdist(material.float(), ALLOYS_SCALED).argmin(1)
            material = ALLOYS_SCALED[alloy]
            material_raw = ALLOYS_RAW[alloy.cpu().numpy()]
        else:
            material_raw = scaler_material.inverse_transform(material.cpu().numpy())

        # 4.-5. re-encode structure + material, predict from the encoder mean
        C = predict_stiffness(tokens[vi], material[vi], material_raw[vi, 0],
                              [sentences[i] for i in vi])
        errors = relative_C_error(C, np.repeat(target, len(vi), axis=0))
        best = int(np.argmin(errors))
        b = vi[best]

        result = {
            "structure": sentences[b],
            "E": float(f"{material_raw[b, 0]:.4g}"),
            "nu": float(f"{material_raw[b, 1]:.4g}"),
            "achieved_properties": _as_dict(C[best]),
            "relative_error_percent": float(f"{100 * errors[best]:.3g}"),
            "valid_candidates": f"{len(vi)} of {N_CANDIDATES}",
        }
        if alloy is not None:
            result["base_material"] = str(ALLOY_NAMES[int(alloy[b])])
        return result
    except Exception as exc:
        return {"error": str(exc)}


inverse_design_tool = FunctionTool(
    run_inverse_design,
    description=(
        "Inverse design: given the 21 target stiffness components C11 ... C66 "
        "(GPa; components that are zero for the symmetry class must be given "
        "as 0), use the latent diffusion model to propose the truss structure "
        "sentence and base material (E, nu) that best achieve them."
    ),
)


# --------------------------------------------------------------------------
# 8. Plotting tools
# --------------------------------------------------------------------------
OUTPUT_DIR = "outputs"
# All images go to outputs/ (default). With AGENTIC_OUTPUT_SUBFOLDERS=1 they are sorted into
# outputs/unit_cells/ and outputs/spider_plots/.
OUTPUT_SUBFOLDERS = os.getenv("AGENTIC_OUTPUT_SUBFOLDERS", "0") == "1"
UNIT_CELL_DIR = os.path.join(OUTPUT_DIR, "unit_cells") if OUTPUT_SUBFOLDERS else OUTPUT_DIR
SPIDER_DIR = os.path.join(OUTPUT_DIR, "spider_plots") if OUTPUT_SUBFOLDERS else OUTPUT_DIR
for _d in (UNIT_CELL_DIR, SPIDER_DIR):
    os.makedirs(_d, exist_ok=True)


def _output_path(folder, save_path, default_name):
    """Images always go to the output folder; only the file name of a save_path chosen
    by the agent is kept."""
    name = os.path.basename(save_path) if save_path else default_name
    if not name.lower().endswith(".png"):
        name += ".png"
    return os.path.join(folder, name)


def _open_image(path):
    """Open a saved image in the default viewer (best effort, any OS).
    Set AGENTIC_NO_OPEN=1 to only save the images."""
    if os.getenv("AGENTIC_NO_OPEN"):
        return
    try:
        if sys.platform.startswith("win"):
            os.startfile(os.path.abspath(path))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def plot_unit_cell(structure: str, E: float, nu: float,
                   save_path: Optional[str] = None) -> dict:
    """Render the 3D truss unit cell for a structure sentence and base material.

    Colors the cell by the reference material closest to (E, nu). Returns the
    saved image path and the matched material info, or {"error": ...}.
    """
    try:
        points, connectivity, connected, _ = build_lattice(structure)
        color, element, material_name = match_material([E, nu], propertiesFile=REFERENCE_MATERIALS, verbose=False)

        ax = draw_unitcell_bezier_colored(
            points, connectivity, control_point_offset_ratio=0, R=5, color=color)

        save_path = _output_path(UNIT_CELL_DIR, save_path,
                                 "unitcell_%s.png" % structure.strip().replace(" ", "_"))
        ax.figure.savefig(save_path, dpi=600, bbox_inches="tight", transparent=True)
        plt.close(ax.figure)
        _open_image(save_path)

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


def plot_radar(predicted_properties: dict, target_properties: Optional[dict] = None,
               E: Optional[float] = None, nu: Optional[float] = None,
               save_path: Optional[str] = None) -> dict:
    """Draw the spider plot of the 21 stiffness components (GPa) in 5 families
    (normal, shear, normal-normal, normal-shear, shear-shear), each family on
    its own scale.

    Pass only ``predicted_properties`` for a forward-simulation result (single
    polygon). Pass both ``predicted_properties`` and ``target_properties`` for
    an inverse-design comparison (target vs. achieved). Keys: C11 ... C66;
    missing keys are taken as 0. Returns the image path and the mode, or
    {"error": ...}.
    """
    try:
        pred_vec = np.array([float(predicted_properties.get(n, 0.0)) for n in PROPERTY_NAMES])
        target_vec = None
        if target_properties:
            target_vec = np.array([float(target_properties.get(n, 0.0)) for n in PROPERTY_NAMES])

        material_color = None
        if E is not None and nu is not None:
            material_color, _, _ = match_material([E, nu], propertiesFile=REFERENCE_MATERIALS, verbose=False)

        ax = radar_compare_groups_C(target_vec, pred_vec, matColor=material_color)
        mode = "compare" if target_vec is not None else "predicted_only"

        save_path = _output_path(SPIDER_DIR, save_path, "radar_%s.png" % mode)
        ax.figure.savefig(save_path, dpi=600, bbox_inches="tight", transparent=True)
        plt.close(ax.figure)
        _open_image(save_path)

        return {"image_path": save_path, "mode": mode}
    except Exception as exc:
        return {"error": str(exc)}


radar_plot_tool = FunctionTool(
    plot_radar,
    description=(
        "Draw the spider plot of the 21 stiffness components C11 ... C66 (GPa). "
        "Pass only predicted_properties for a forward result; pass both "
        "predicted_properties and target_properties for an inverse-design "
        "target-vs-achieved comparison."
    ),
)


# --------------------------------------------------------------------------
# 9. Agents
# --------------------------------------------------------------------------
# LLM for the orchestrator, the agents and the speaker selection. Pinned to a
# dated snapshot (the alias "gpt-4o" moves to newer snapshots over time);
# default sampling settings of the API (temperature 1). Override with AGENTIC_LLM.
LLM_MODEL = os.getenv("AGENTIC_LLM", "gpt-4o-2024-08-06")
try:                                    # models AutoGen knows (e.g. gpt-4o, gpt-4.1)
    from autogen_ext.models.openai._model_info import get_info
    get_info(LLM_MODEL)
    _model_info = None
except Exception:                       # newer models (e.g. gpt-5.2, gpt-5.4, gpt-5.5): declare capabilities
    _model_info = {"vision": False, "function_calling": True, "json_output": True,
                   "structured_output": True, "family": "unknown", "multiple_system_messages": True}
# Reasoning models (e.g. gpt-6-sol, gpt-5.6-*) accept function tools on the chat-completions endpoint only with
# reasoning switched off: set AGENTIC_REASONING_EFFORT=none for them.
_extra = {"reasoning_effort": os.getenv("AGENTIC_REASONING_EFFORT")} if os.getenv("AGENTIC_REASONING_EFFORT") else {}
model_client = OpenAIChatCompletionClient(model=LLM_MODEL, api_key=API_KEY, model_info=_model_info, **_extra)

# Speaker selection. "rules" (default): the next agent follows the workflow
# deterministically (planner -> the agent it delegated to -> the matching
# plotting agent, or back to the planner if the tool returned an error ->
# planner); the LLM selector is used only when the planner names no agent.
# "llm": every speaker is chosen by the LLM with selector_prompt. With "llm",
# the selector can skip the simulation/design agent and hand the request to a
# plotting agent, which then has no tool output to plot. "rules" also tells the
# plotting agents never to plot numbers that no tool returned ("llm" keeps the
# original prompts).
ROUTING = os.getenv("AGENTIC_ROUTING", "rules")
PLOT_GUARD = ("""
Only plot numbers that a tool returned earlier in this conversation; never
invent or estimate values. If no tool result is available (or the tool
returned an error), do not call any tool: say that there is nothing to plot.
""" if ROUTING == "rules" else "")
PLANNER_ERROR_RULE = ("""
  - If SimulationAgent or DesignAgent reports an error, do not delegate to a
    plotting agent: explain the error and how to fix the input, and end with
    "TERMINATE".""" if ROUTING == "rules" else "")

STIFFNESS_NOTE = """Stiffness components (GPa), upper triangle of the 6x6 matrix, Voigt order
1=11, 2=22, 3=33, 4=12, 5=13, 6=23:
  C11 C22 C33         normal stiffness
  C44 C55 C66         shear stiffness (planes 1-2, 1-3, 2-3)
  C12 C13 C23         normal-normal coupling
  C14 ... C36         normal-shear coupling (C14 C24 C34 monoclinic, C16 C26 C36 trigonal)
  C45 C46 C56         shear-shear coupling (C56 monoclinic, C45 trigonal)
  C15 C25 C35 C46 are zero for every symmetry class."""

simulation_agent = AssistantAgent(
    name="SimulationAgent",
    description=("Forward model: given a structure sentence and base material "
                 "(E, nu), predicts the 21 stiffness components C11 ... C66."),
    model_client=model_client,
    tools=[simulation_tool],
    reflect_on_tool_use=True,
    system_message=f"""
You are the Simulation Agent for a truss metamaterial framework.

Tool: run_simulation(structure, E, nu) -> 21 predicted stiffness components
{STIFFNESS_NOTE}

Steps:
1. Call run_simulation with the user inputs.
2. Present the result as a Component | Value (GPa) table.
3. Give a short physical reading (anisotropy, stiffest direction, shear vs.
   normal stiffness, notable couplings).
4. On error, explain it and tell the user how to fix the input.
""",
)

design_agent = AssistantAgent(
    name="DesignAgent",
    description=("Inverse model: given 21 target stiffness components, proposes "
                 "the truss structure and base material that best achieve them."),
    model_client=model_client,
    tools=[inverse_design_tool],
    reflect_on_tool_use=True,
    system_message=f"""
You are the Design Agent for a truss metamaterial framework.

Tool: run_inverse_design(21 target stiffness components C11 ... C66, GPa)
  returns: structure sentence, base material (E, nu; plus base_material, the
  alloy name, when the material database is available), achieved_properties
  and relative_error_percent (||C - C_target|| / ||C_target||).
{STIFFNESS_NOTE}
Components the user does not give are 0.

Steps:
1. Call run_inverse_design with the target components.
2. Present the recommended structure sentence and base material (E, nu, and
   the alloy name if returned).
3. Show a Component | Target | Achieved table for the non-zero components.
4. Report the relative error and flag notable deviations.
5. On error, explain it and ask the user to check the inputs.
""",
)

sim_plotting_agent = AssistantAgent(
    name="SimPlottingAgent",
    description=("Visualizes a forward result: draws only the predicted-stiffness "
                 "spider plot (no unit cell, no target)."),
    model_client=model_client,
    tools=[radar_plot_tool],
    reflect_on_tool_use=True,
    system_message=f"""
You visualize forward-simulation results.
{PLOT_GUARD}
Tool: plot_radar. Call it with predicted_properties set to the dictionary
returned by run_simulation, unchanged (a JSON object with the keys C11 ... C66,
not a list), and pass E and nu for coloring. Never pass
target_properties -- a forward run has no target. Report the image path.
""",
)

design_plotting_agent = AssistantAgent(
    name="DesignPlottingAgent",
    description=("Visualizes an inverse result: draws the 3D unit cell and a "
                 "target-vs-achieved stiffness spider plot."),
    model_client=model_client,
    tools=[unit_cell_plot_tool, radar_plot_tool],
    reflect_on_tool_use=True,
    system_message=f"""
You visualize inverse-design results.
{PLOT_GUARD}
Tools: plot_unit_cell and plot_radar.

Steps:
1. Call plot_unit_cell with the structure sentence and base material (E, nu)
   from DesignAgent.
2. Call plot_radar with predicted_properties = achieved_properties (the
   dictionary returned by run_inverse_design, unchanged), target_properties =
   the user's original target components as a JSON object with the keys
   C11 ... C66 (missing components are 0; never a list), and E/nu for coloring.
3. Report both image paths and the matched material (element, name, color).
   Ensure transparent background at 600 DPI.
""",
)

planning_agent = AssistantAgent(
    name="PlanningAgent",
    description=("Orchestrator: routes the request to the Simulation or Design "
                 "agent, then to the matching plotting agent, then summarizes."),
    model_client=model_client,
    system_message=f"""
You are the orchestrating agent for a truss metamaterial framework.

Team:
  SimulationAgent     forward: structure + material -> 21 stiffness components
  DesignAgent         inverse: 21 target stiffness components -> structure + material
  SimPlottingAgent    spider plot for a forward result
  DesignPlottingAgent unit cell + target-vs-achieved spider plot for an inverse result

Rules:
  - If the user gives a structure + material and wants properties -> SimulationAgent.
  - If the user gives target stiffness components and wants a design -> DesignAgent.
  - Delegate with:  1. <AgentName> : <task with all inputs>
  - After SimulationAgent, delegate to SimPlottingAgent with the structure,
    material (E, nu), and predicted components.
  - After DesignAgent, delegate to DesignPlottingAgent with the structure,
    material (E, nu), achieved components, and original targets.
  - You never call tools yourself.
  - After the plotting agent reports, summarize in plain language and end with
    "TERMINATE".{PLANNER_ERROR_RULE}
""",
)


# --------------------------------------------------------------------------
# 10. Team
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



def _last_tool_failed(messages, agent):
    """True if the last tool result of ``agent`` is missing or an error."""
    for m in reversed(messages):
        if getattr(m, "source", None) == agent and type(m).__name__ == "ToolCallExecutionEvent":
            return any(r.is_error or "'error'" in str(r.content) for r in m.content)
    return True


def rule_based_selector(messages):
    last = messages[-1]
    source = getattr(last, "source", None)
    if source == "user":
        return "PlanningAgent"
    if source == "PlanningAgent":
        text = str(getattr(last, "content", ""))
        named = [a for a in ("DesignAgent", "SimulationAgent") if a in text]
        return named[0] if len(named) == 1 else None          # None: LLM selector decides
    if source == "SimulationAgent":
        return "PlanningAgent" if _last_tool_failed(messages, source) else "SimPlottingAgent"
    if source == "DesignAgent":
        return "PlanningAgent" if _last_tool_failed(messages, source) else "DesignPlottingAgent"
    if source in ("SimPlottingAgent", "DesignPlottingAgent"):
        return "PlanningAgent"
    return None


team = SelectorGroupChat(
    participants=[planning_agent, simulation_agent, design_agent,
                  sim_plotting_agent, design_plotting_agent],
    model_client=model_client,
    termination_condition=termination,
    selector_prompt=selector_prompt,
    selector_func=rule_based_selector if ROUTING == "rules" else None,
    allow_repeated_speaker=True,
)


# --------------------------------------------------------------------------
# 11. Entry point
# --------------------------------------------------------------------------
async def main():
    from autogen_agentchat.ui import Console

    # Forward-task example:
    task = os.getenv("AGENTIC_TASK") or ("Run a simulation for the structure 'Orthotropic v3 e0 e2 e5 e10 f0' "
            "with E=210.0 and nu=0.3.")

    # Inverse-task example (a cubic test target; all other components are 0):
    #task = ("I need a truss design with the stiffness components (GPa) "
    #        "C11=0.0529, C22=0.0529, C33=0.0529, C12=0.00876, C13=0.00876, "
    #        "C23=0.00876, C44=0.0584, C55=0.0584, C66=0.0584; all other "
    #        "components are 0.")

    await Console(team.run_stream(task=task))


if __name__ == "__main__":
    asyncio.run(main())
