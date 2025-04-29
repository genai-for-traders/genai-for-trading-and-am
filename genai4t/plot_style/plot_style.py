# plot_style.py
import matplotlib.pyplot as plt

# Font settings
FONT_FAMILY = "Times New Roman"
TITLE_SIZE = 18
LABEL_SIZE = 16
TICK_SIZE = 14

# Default Figure Sizes
DEFAULT_FIGSIZE = (8, 6)   # Single-column figure
WIDE_FIGSIZE = (10, 6)     # Double-column figure
COMPACT_FIGSIZE = (6, 4)   # For small panel-based figures

def get_figsize(size="default"):
    """
    Returns the appropriate figure size based on the selected type.
    
    - "default"  -> (8,6)  [Single-column standard]
    - "wide"     -> (10,6) [Double-column layout]
    - "compact"  -> (6,4)  [Smaller figures, multi-panel layouts]
    - Custom     -> Pass a tuple, e.g., get_figsize((12, 6))
    """
    sizes = {
        "default": DEFAULT_FIGSIZE,
        "wide": WIDE_FIGSIZE,
        "compact": COMPACT_FIGSIZE
    }
    return sizes.get(size, size)  # If a custom tuple is passed, return that

def apply_plot_style(ax, boxed=True, tight_layout=True):
    """
    Apply consistent font and axis styling to a given matplotlib axis.
    
    Parameters:
    - ax: The axis object to style.
    - boxed: If True, enables a box frame around the plot (all spines visible).
    """
    ax.set_xlabel(ax.get_xlabel(), fontsize=LABEL_SIZE, fontname=FONT_FAMILY, fontweight='normal')
    ax.set_ylabel(ax.get_ylabel(), fontsize=LABEL_SIZE, fontname=FONT_FAMILY, fontweight='normal')
    ax.set_title(ax.get_title(), fontsize=TITLE_SIZE, fontname=FONT_FAMILY, fontweight='normal')

    # Set tick labels
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname(FONT_FAMILY)

    # Spine (border) styling
    if boxed:
        # Keep all spines visible
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)  # Slightly thicker for emphasis
    else:
        # Hide top and right spines for a cleaner look
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Apply tight layout if requested
    if tight_layout:
        ax.get_figure().tight_layout()

def apply_grid(ax, axis="y", linestyle="--", alpha=0.6):
    """
    Apply a grid selectively to an axis (x or y).
    """
    ax.grid(axis=axis, linestyle=linestyle, alpha=alpha)
