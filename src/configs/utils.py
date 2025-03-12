from typing import Dict, Any
from flax import struct


def flax_struct_to_dict(target):
    target_dict = {}
    for field_info in struct.dataclasses.fields(target):
        target_dict[field_info.name] = getattr(target, field_info.name)
    return target_dict


# this function is copied from https://gist.github.com/Microsheep/11edda9dee7c1ba0c099709eb7f8bea7
def transform_dict(config_dict: Dict, expand: bool = True):
    """
    General function to transform any dictionary into wandb config acceptable format
    (This is mostly due to datatypes that are not able to fit into YAML format which makes wandb angry)
    The expand argument is used to expand iterables into dictionaries so that these configs can be used when compare across runs
    """
    ret: Dict[str, Any] = {}
    for k, v in config_dict.items():
        if v is None or isinstance(v, (int, float, str)):
            ret[k] = v
        elif isinstance(v, (list, tuple, set)):
            # Need to check if item in iterable is YAML-friendly
            t = transform_dict(dict(enumerate(v)), expand)
            # Transform back to iterable if expand is False
            ret[k] = t if expand else [t[i] for i in range(len(v))]
        elif isinstance(v, dict):
            ret[k] = transform_dict(v, expand)
        else:
            # Transform to YAML-friendly (str) format
            # Need to handle both Classes, Callables, Object Instances
            # Custom Classes might not have great __repr__ so __name__ might be better in these cases
            vname = v.__name__ if hasattr(v, "__name__") else v.__class__.__name__
            ret[k] = f"{v.__module__}:{vname}"
    return ret
