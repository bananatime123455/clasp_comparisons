import os

MOD_MAP = {
    'T1_Post': 't1post',
    'T1_Pre':  't1pre',
    'T2':      't2',
    'FLAIR':   'flair',
}

def get_checkpoint(mod1: str, mod2: str, checkpoint_dir: str = './models/Clasp_2D/model_checkpoints') -> str:
    key1 = MOD_MAP.get(mod1)
    key2 = MOD_MAP.get(mod2)

    if key1 is None or key2 is None:
        raise ValueError(f"Unknown modality: {mod1!r} or {mod2!r}. "
                         f"Valid options: {list(MOD_MAP.keys())}")

    # Sort so t1post_and_t2 == t2_and_t1post
    key1, key2 = sorted([key1, key2])
    filename   = f"{key1}_and_{key2}.ckpt"
    checkpoint = os.path.join(checkpoint_dir, filename)

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"No checkpoint found at: {checkpoint}")

    return checkpoint