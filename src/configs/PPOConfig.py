from .utils import transform_dict, flax_struct_to_dict
from flax import struct
import typing as t


@struct.dataclass
class PPOConfig:
    gamma: float = struct.field(pytree_node=False)
    gae_lambda: float = struct.field(pytree_node=False)
    rollout_steps: int = struct.field(pytree_node=False)
    total_steps: int = struct.field(pytree_node=False)
    epochs: int = struct.field(pytree_node=False)
    num_updates: int = struct.field(pytree_node=False)
    num_mini_batch: int = struct.field(pytree_node=False)
    clip_eps: float = struct.field(pytree_node=False)
    vf_coef: float = struct.field(pytree_node=False)
    max_grad_norm: float = struct.field(pytree_node=False)
    lr: float = struct.field(pytree_node=False)
    rec_fn: str = struct.field(pytree_node=False)
    activation: str = struct.field(pytree_node=False)
    seq_len_in_minibatch: int = struct.field(pytree_node=False)
    entropy_coef: float = struct.field(pytree_node=False)
    gradient_clipping: bool = struct.field(pytree_node=False)
    stale_gradient: bool = struct.field(pytree_node=False)
    stale_target: bool = struct.field(pytree_node=False)
    grad_estimator: str = struct.field(pytree_node=False)
    d_shared_repr: list[int] = struct.field(pytree_node=False)
    d_rec: int = struct.field(pytree_node=False)
    d_actor_head: int = struct.field(pytree_node=False)
    d_critic_head: int = struct.field(pytree_node=False)

    @classmethod
    def from_dict(cls: t.Type["PPOConfig"], obj: dict):
        return cls(
            gamma=obj["gamma"],
            gae_lambda=obj["gae_lambda"],
            rollout_steps=obj["rollout_steps"],
            total_steps=obj["total_steps"],
            epochs=obj["epochs"],
            num_updates=obj.get(
                "num_updates", obj["total_steps"] // obj["rollout_steps"]
            ),
            num_mini_batch=obj["num_mini_batch"],
            clip_eps=obj["clip_eps"],
            vf_coef=obj["vf_coef"],
            max_grad_norm=obj["max_grad_norm"],
            seq_len_in_minibatch=obj["seq_len_in_minibatch"],
            lr=obj["lr"],
            rec_fn=obj["rec_fn"],
            activation=obj["activation"],
            entropy_coef=obj["entropy_coef"],
            gradient_clipping=obj["gradient_clipping"],
            stale_gradient=obj.get("stale_gradient", True),
            stale_target=obj.get("stale_target", True),
            grad_estimator=obj.get("grad_estimator", "bp"),
            d_shared_repr=obj["d_shared_repr"],
            d_rec=obj["d_rec"],
            d_actor_head=obj["d_actor_head"],
            d_critic_head=obj["d_critic_head"],
        )

    def to_dict(self, expand: bool = True):
        state_dict = flax_struct_to_dict(self)
        return transform_dict(state_dict, expand)
