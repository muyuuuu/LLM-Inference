import torch


class _RMSNorm:
    def __init__(self, weight, eps=1e-5, device="cpu"):
        self._device = device
        self._eps = torch.tensor(eps, device=device)
        self._weight = weight

    def __call__(self, x):
        ori_type = x.dtype
        x = x.to(torch.float32)

        y = x / torch.sqrt(torch.mean(x**2, -1, keepdim=True) + self._eps)
        y = self._weight * y
        y = y.to(ori_type)

        return y
