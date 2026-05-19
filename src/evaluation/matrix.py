# ======================================
# CONFUSION MATRIX GENERATOR
# ======================================

import os
import json
import textwrap

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

# ======================================
# LOAD RESULTS FUNCTION
# ======================================

def load_results(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return pd.DataFrame(data)


# ======================================
# EXPERIMENT DEFINITIONS
# ======================================

experiments = [

    # ==================================
    # CLAUDE — WITH CONTEXT
    # ==================================

    {
        "model": "claude_sonnet",
        "task": "cps_behavior",
        "context": True,
        "path": "src/results/raw/claude_cps_behavior_with_context.json"
    },

    {
        "model": "claude_sonnet",
        "task": "interactional_move",
        "context": True,
        "path": "src/results/raw/claude_interactional_move_with_context.json"
    },

    {
        "model": "claude_sonnet",
        "task": "is_followup",
        "context": True,
        "path": "src/results/raw/claude_is_followup_with_context.json"
    },

    {
        "model": "claude_sonnet",
        "task": "prompt_type",
        "context": True,
        "path": "src/results/raw/claude_prompt_type_with_context.json"
    },

    # ==================================
    # GPT-4o — WITH CONTEXT
    # ==================================

    {
        "model": "gpt4o",
        "task": "cps_behavior",
        "context": True,
        "path": "src/results/raw/gpt4o_cps_behavior_with_context.json"
    },

    {
        "model": "gpt4o",
        "task": "interactional_move",
        "context": True,
        "path": "src/results/raw/gpt4o_interactional_move_with_context.json"
    },

    {
        "model": "gpt4o",
        "task": "is_followup",
        "context": True,
        "path": "src/results/raw/gpt4o_is_followup_with_context.json"
    },

    {
        "model": "gpt4o",
        "task": "prompt_type",
        "context": True,
        "path": "src/results/raw/gpt4o_prompt_type_with_context.json"
    },

    # ==================================
    # CLAUDE — WITHOUT CONTEXT
    # ==================================

    {
        "model": "claude_sonnet",
        "task": "cps_behavior",
        "context": False,
        "path": "src/results/raw/claude_cps_behavior.json"
    },

    {
        "model": "claude_sonnet",
        "task": "interactional_move",
        "context": False,
        "path": "src/results/raw/claude_interactional_move.json"
    },

    {
        "model": "claude_sonnet",
        "task": "is_followup",
        "context": False,
        "path": "src/results/raw/claude_is_followup.json"
    },

    {
        "model": "claude_sonnet",
        "task": "prompt_type",
        "context": False,
        "path": "src/results/raw/claude_prompt_type.json"
    },

    # ==================================
    # GPT-4o — WITHOUT CONTEXT
    # ==================================

    {
        "model": "gpt4o",
        "task": "cps_behavior",
        "context": False,
        "path": "src/results/raw/gpt4o_cps_behavior.json"
    },

    {
        "model": "gpt4o",
        "task": "interactional_move",
        "context": False,
        "path": "src/results/raw/gpt4o_interactional_move.json"
    },

    {
        "model": "gpt4o",
        "task": "is_followup",
        "context": False,
        "path": "src/results/raw/gpt4o_is_followup.json"
    },

    {
        "model": "gpt4o",
        "task": "prompt_type",
        "context": False,
        "path": "src/results/raw/gpt4o_prompt_type.json"
    },

    # ==================================
    # REFLEKTOBOT ONLY — INTERACTIONAL MOVE
    # ==================================

    {
        "model": "claude_sonnet",
        "task": "interactional_move",
        "context": True,
        "assistant_name": "ReflektoBot",
        "path": "src/results/raw/claude_interactional_move_with_context.json"
    },

    {
        "model": "gpt4o",
        "task": "interactional_move",
        "context": True,
        "assistant_name": "ReflektoBot",
        "path": "src/results/raw/gpt4o_interactional_move_with_context.json"
    },

    {
        "model": "claude_sonnet",
        "task": "interactional_move",
        "context": False,
        "assistant_name": "ReflektoBot",
        "path": "src/results/raw/claude_interactional_move.json"
    },

    {
        "model": "gpt4o",
        "task": "interactional_move",
        "context": False,
        "assistant_name": "ReflektoBot",
        "path": "src/results/raw/gpt4o_interactional_move.json"
    },

    # ==================================
    # IS_FOLLOWUP — SEPARATED BY ROLE
    # ==================================

    {
        "model": "claude_sonnet",
        "task": "is_followup",
        "context": True,
        "role_filter": 0,
        "path": "src/results/raw/claude_is_followup_with_context.json"
    },

    {
        "model": "claude_sonnet",
        "task": "is_followup",
        "context": True,
        "role_filter": 1,
        "path": "src/results/raw/claude_is_followup_with_context.json"
    },

    {
        "model": "gpt4o",
        "task": "is_followup",
        "context": True,
        "role_filter": 0,
        "path": "src/results/raw/gpt4o_is_followup_with_context.json"
    },

    {
        "model": "gpt4o",
        "task": "is_followup",
        "context": True,
        "role_filter": 1,
        "path": "src/results/raw/gpt4o_is_followup_with_context.json"
    },

    {
        "model": "claude_sonnet",
        "task": "is_followup",
        "context": False,
        "role_filter": 0,
        "path": "src/results/raw/claude_is_followup.json"
    },

    {
        "model": "claude_sonnet",
        "task": "is_followup",
        "context": False,
        "role_filter": 1,
        "path": "src/results/raw/claude_is_followup.json"
    },

    {
        "model": "gpt4o",
        "task": "is_followup",
        "context": False,
        "role_filter": 0,
        "path": "src/results/raw/gpt4o_is_followup.json"
    },

    {
        "model": "gpt4o",
        "task": "is_followup",
        "context": False,
        "role_filter": 1,
        "path": "src/results/raw/gpt4o_is_followup.json"
    }

]
# ======================================
# OUTPUT DIRECTORY
# ======================================

OUTPUT_DIR = "src/evaluation/confusion_matrices"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ======================================
# PLOTTING SETTINGS
# ======================================

sns.set_theme(style="white")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 11


# ======================================
# HELPER FUNCTIONS
# ======================================

def prettify_label(label: str) -> str:
    """
    Makes labels more readable for plots.
    Example:
        negotiation_coordination -> Negotiation Coordination
    """
    return label.replace("_", " ").title()


def wrap_labels(labels, width=18):
    """
    Wrap long labels into multiple lines.
    """
    return [
        "\n".join(textwrap.wrap(label, width))
        for label in labels
    ]


def create_confusion_matrix_plot(
    df: pd.DataFrame,
    title: str,
    save_path: str,
    assistant_name: str = None,
    role_filter: int = None
):
    """
    Creates and saves a confusion matrix heatmap.
    
    Args:
        df: DataFrame with results
        title: Title for the plot
        save_path: Path to save the plot
        assistant_name: Optional filter for specific assistant (e.g., "ReflektoBot")
        role_filter: Optional filter by role (0=teacher, 1=bot)
    """

    # ==================================
    # FILTER BY ASSISTANT NAME IF PROVIDED
    # ==================================

    if assistant_name:
        df = df[df["assistant_name"] == assistant_name].copy()
        if df.empty:
            print(f"Skipping empty dataset after filtering for {assistant_name}")
            return

    # ==================================
    # FILTER BY ROLE IF PROVIDED
    # ==================================

    if role_filter is not None:
        df = df[df["role"] == role_filter].copy()
        if df.empty:
            print(f"Skipping empty dataset after filtering for role {role_filter}")
            return

    # ==================================
    # NORMALIZE LABELS
    # ==================================

    df["true_label"] = (
        df["true_label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["predicted_label"] = (
        df["predicted_label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # ==================================
    # LABELS
    # ==================================

    labels = sorted(
        list(
            set(df["true_label"].unique())
            | set(df["predicted_label"].unique())
        )
    )

    # ==================================
    # CONFUSION MATRIX
    # ==================================

    cm = confusion_matrix(
    df["true_label"],
    df["predicted_label"],
    labels=labels
    )

    # ==================================
    # PRETTY LABELS
    # ==================================

    pretty_labels = [
        prettify_label(label)
        for label in labels
    ]

    wrapped_labels = wrap_labels(pretty_labels)

    # ==================================
    # FIGURE SIZE
    # Dynamically scale based on number of labels
    # ==================================

    size = max(8, len(labels) * 1.2)

    # ==================================
    # FIGURE
    # ==================================

    fig, ax = plt.subplots(
        figsize=(size, size + 1.5)
    )

    # ==================================
    # HEATMAP
    # ==================================

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.5,
        linecolor="lightgray",
        square=True,
        cbar=True,
        xticklabels=wrapped_labels,
        yticklabels=wrapped_labels,
        annot_kws={"size": 10},
        cbar_kws={
            "shrink": 0.85,
            "pad": 0.04
        },
        ax=ax
    )

    # ==================================
    # AXIS LABELS
    # ==================================

    ax.set_xlabel(
        "Predicted Label",
        fontsize=14,
        fontweight="bold",
        labelpad=20
    )

    ax.set_ylabel(
        "Human Annotated True Label",
        fontsize=14,
        fontweight="bold",
        labelpad=20
    )

    # ==================================
    # TITLE
    # ==================================

    ax.set_title(
        title,
        fontsize=18,
        fontweight="bold",
        pad=40
    )

    # ==================================
    # TICKS
    # ==================================

    plt.xticks(
        rotation=45,
        ha="right",
        fontsize=10
    )

    plt.yticks(
        rotation=0,
        fontsize=10
    )

    # ==================================
    # SPACING
    # ==================================

    fig.subplots_adjust(
        top=0.88,
        right=0.92,
        left=0.22,
        bottom=0.22
    )

    # ==================================
    # SAVE
    # ==================================

    plt.savefig(
        save_path,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close()

    print(f"Saved: {save_path}")


# ======================================
# GENERATE MATRICES
# ======================================

for experiment in experiments:

    # ==================================
    # LOAD DATA
    # ==================================

    df = load_results(experiment["path"])

    # ==================================
    # CREATE CONTEXT LABEL
    # ==================================

    context_label = (
        "with_context"
        if experiment["context"]
        else "without_context"
    )

    # ==================================
    # GET OPTIONAL ASSISTANT NAME FILTER
    # ==================================

    assistant_name = experiment.get("assistant_name", None)
    assistant_label = (
        f"_{assistant_name}"
        if assistant_name
        else ""
    )

    # ==================================
    # GET OPTIONAL ROLE FILTER
    # ==================================

    role_filter = experiment.get("role_filter", None)
    role_label_map = {0: "_teacher", 1: "_bot"}
    role_label = (
        role_label_map.get(role_filter, "")
        if role_filter is not None
        else ""
    )

    # ==================================
    # GENERATE MATRICES FOR EACH STRATEGY
    # ==================================

    for strategy in ["basic", "few_shot"]:

        strategy_df = df[
            df["strategy"] == strategy
        ].copy()

        # Skip empty data
        if strategy_df.empty:
            print(
                f"Skipping empty dataset: "
                f"{experiment['task']} - {strategy}"
            )
            continue

        # ==================================
        # TITLE
        # ==================================

        title_parts = [
            experiment['model'],
            experiment['task'],
            strategy.replace('_', ' ').title(),
            context_label.replace('_', ' ').title()
        ]
        
        if assistant_name:
            title_parts.append(f"({assistant_name})")
        
        if role_filter is not None:
            role_text = "Teacher" if role_filter == 0 else "Bot"
            title_parts.append(f"({role_text})")
        
        title = " | ".join(title_parts)

        # ==================================
        # FILE NAME
        # ==================================

        filename = (
            f"{experiment['model']}_"
            f"{experiment['task']}_"
            f"{strategy}_"
            f"{context_label}"
            f"{assistant_label}"
            f"{role_label}.png"
        )

        save_path = os.path.join(
            OUTPUT_DIR,
            filename
        )

        # ==================================
        # CREATE MATRIX
        # ==================================

        create_confusion_matrix_plot(
            df=strategy_df,
            title=title,
            save_path=save_path,
            assistant_name=assistant_name,
            role_filter=role_filter
        )


print("\nAll confusion matrices generated successfully.")