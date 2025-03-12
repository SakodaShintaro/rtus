from .utils import transform_dict, flax_struct_to_dict
from flax import struct
import typing as t
from .PPOConfig import PPOConfig
from .EnvConfig import EnvConfig
import yaml


@struct.dataclass
class ExpConfig:
    tag: str = struct.field(
        pytree_node=False
    )  # a tag to identify the experiment in wandb
    wandb_project_name: str = struct.field(pytree_node=False)
    exp_seed: int = struct.field(pytree_node=False)
    ppo_config: PPOConfig = struct.field(pytree_node=False)
    env_config: EnvConfig = struct.field(pytree_node=False)

    @classmethod
    def from_dict(cls: t.Type["EnvConfig"], obj: dict):
        return cls(
            tag=obj["tag"],
            wandb_project_name=obj["wandb_project_name"],
            exp_seed=obj["exp_seed"],
            ppo_config=PPOConfig.from_dict(obj["ppo_config"]),
            env_config=EnvConfig.from_dict(obj["env_config"]),
        )

    def to_dict(self, expand: bool = True):
        ppo_config_dict = flax_struct_to_dict(self.ppo_config)
        env_config_dict = flax_struct_to_dict(self.env_config)
        exp_dict = flax_struct_to_dict(self)
        exp_dict["ppo_config"] = ppo_config_dict
        exp_dict["env_config"] = env_config_dict
        return transform_dict(exp_dict, expand)

    def from_yaml(config_file="configs/gymnax_config.yaml"):
        with open(config_file, "r") as stream:
            config = yaml.safe_load(stream)
        config = ExpConfig.from_dict(config)
        return config
