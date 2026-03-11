from datasets import load_dataset, load_from_disk
import os

def load_mini_imagenet(subset=None, cache_path=None):

    if cache_path and os.path.exists(cache_path):
        ds = load_from_disk(cache_path)
    else:
        ds = load_dataset("timm/mini-imagenet")
        if cache_path:
            ds.save_to_disk(cache_path)

    if subset is not None:
        ds = {k: v.select(range(min(subset, len(v)))) for k, v in ds.items()}

    return ds