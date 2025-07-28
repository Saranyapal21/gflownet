# gflownet/proxy/frozenlake.py
from typing import Optional
import torch
from torchtyping import TensorType
from gflownet.proxy.base import Proxy
from gflownet.utils.common import tfloat

# In gflownet/proxy/frozenlake.py
HOLES_4x4 = [5, 7, 11, 12]
GOAL_4x4  = 15

hole_E = 5.0     # high energy  -> tiny reward
goal_E = 0.0     # low energy   -> large reward



class FrozenLakeProxy(Proxy):
    def __init__(self, map_name: str = "4x4", eps: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        assert map_name == "4x4", "FrozenLakeProxy currently supports only 4x4"
        self.eps = float(eps)

        # build reward vector in the framework's default float dtype
        # rvec = torch.full((16,), self.eps, dtype=self.float)
        # rvec[HOLES_4x4] = self.eps
        # rvec[GOAL_4x4]  = 1.0


        #   Did some more improvements for better training (Before this, GFn focused on holes in the env as holes have lower energy.... Reward -> energy)

        rvec = torch.full((16,), hole_E, dtype=self.float)
        rvec[GOAL_4x4] = goal_E
        self.rvec = rvec.to(self.device)


    def __call__(self, states_proxy: TensorType["batch", 16]) -> TensorType["batch"]:
        # cast inputs to our device; tfloat doesn’t take dtype
        x = tfloat(states_proxy, self.device, self.float)
        # ensure dtype matches rvec to avoid matmul dtype mismatch
        if x.dtype != self.rvec.dtype:
            x = x.to(self.rvec.dtype)
        return x @ self.rvec

    def to(self, device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None):
        if device is not None:
            self.device = device
        if dtype is not None:
            self.float = dtype
            self.rvec = self.rvec.to(dtype=dtype)
        self.rvec = self.rvec.to(self.device)
        return self
