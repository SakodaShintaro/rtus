from .utils import transform_dict, flax_struct_to_dict
from flax import struct
import typing as t


@struct.dataclass
class EnvConfig:
    # use pytree_node=False to indicate an attribute should not be touched
    # by Jax transformations.i.e, static values
    env_name: str = struct.field(pytree_node=False)
    domain: str = struct.field(pytree_node=False)
    normalize_obs: bool = struct.field(pytree_node=False)
    normalize_reward: bool = struct.field(pytree_node=False)
    clip_action: bool = struct.field(pytree_node=False)
    continous_action: bool = struct.field(pytree_node=False)
    add_reward_prev_action: bool = struct.field(pytree_node=False)

    @classmethod
    def from_dict(cls: t.Type["EnvConfig"], obj: dict):
        return cls(
            env_name=obj["env_name"],
            domain=obj["domain"],
            normalize_obs=obj.get("normalize_obs", False),
            normalize_reward=obj.get("normalize_reward", False),
            clip_action=obj.get("clip_action", False),
            continous_action=obj.get("continous_a", False),
            add_reward_prev_action=obj.get("add_reward_prev_action", False),
        )

    def to_dict(self, expand: bool = True):
        state_dict = flax_struct_to_dict(self)
        return transform_dict(state_dict, expand)
