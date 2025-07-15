import functools
import os
import random
from pathlib import Path
import inflect

# Initialize inflect engine
IE = inflect.engine()

ASSETS_PATH = Path(__file__).parent / "assets"

if not ASSETS_PATH.exists():
    raise FileNotFoundError(f"Assets directory not found at {ASSETS_PATH}")

@functools.cache
def _load_lines(path):
    """
    Load lines from a file. First tries to load from `path` directly, and if that doesn't exist, searches the
    `assets` directory for a file named `path`.
    """
    file_path = Path(path)

    if not file_path.exists():
        file_path = ASSETS_PATH / path

    if not file_path.exists():
        raise FileNotFoundError(f"Could not find {path} or assets/{path}")

    with file_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]
        
@functools.cache
def _split_train_val(path, val_frac=0.1):
    """
    Split the simple_animals lines into training and validation sets based on val_frac.

    Args:
        val_frac (float): Fraction of data to be used for validation.

    Returns:
        tuple: (train_lines, val_lines)
    """
    if not 0.0 <= val_frac <= 1.0:
        raise ValueError("val_frac must be between 0.0 and 1.0")

    lines = _load_lines(path)
    if not lines:
        raise ValueError(f"The file {path} is empty.")
    lines = _load_lines(path)
    shuffled = lines.copy()
    random.shuffle(shuffled)
    split_index = int(len(shuffled) * (1 - val_frac))
    train_lines = shuffled[:split_index]
    val_lines = shuffled[split_index:]
    return train_lines, val_lines


def from_file(path, low=None, high=None):
    prompts = _load_lines(path)[low:high] 
    return random.choice(prompts), {}

def hps_v2_all():
    return from_file("hps_v2_all.txt")

def simple_animals():
    return from_file("simple_animals.txt")

def eval_simple_animals():
    return from_file("eval_simple_animals.txt")

def eval_hps_v2_all():
    return from_file("hps_v2_all_eval.txt")

def simple_animals(val_frac=0.1):
    """
    Split the 'simple_animals.txt' dataset into training and validation sets and return corresponding functions.

    Args:
        val_frac (float, optional): Fraction of data to be used for validation. Defaults to 0.1.

    Returns:
        tuple: (train_function, val_function)
    """
    if val_frac>0:
        train_lines, val_lines = _split_train_val("simple_animals.txt", val_frac)

        def train():
            """
            Get a random training prompt from 'simple_animals.txt'.

            Returns:
                tuple: (prompt, metadata)
            """
            if not train_lines:
                raise ValueError("Training set is empty. Adjust val_frac.")
            return random.choice(train_lines), {}

        def val():
            """
            Get a random validation prompt from 'simple_animals.txt'.

            Returns:
                tuple: (prompt, metadata)
            """
            if not val_lines:
                raise ValueError("Validation set is empty. Adjust val_frac.")
            return random.choice(val_lines), {}

        return train, val
    else:
        def train():
            """
            Get a random training prompt from 'simple_animals.txt'.

            Returns:
                tuple: (prompt, metadata)
            """
            return from_file("simple_animals.txt")
        return train, None
