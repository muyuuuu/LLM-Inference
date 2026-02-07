import torch


class EasyKVCache:
    def __init__(self):
        self._offset = 0
        self._cached_kv = (None, None)

    def save_kv_to_cache(self, k_tensor, v_tensor):
        assert (
            k_tensor.dim() == v_tensor.dim() == 4
        ), "k v tensor shape must be [batch, num_head, seq_len, head_dim]"

        if self._cached_kv[0] is None:
            self._cached_kv = (k_tensor, v_tensor)
        else:
            new_k = torch.cat([self._cached_kv[0], k_tensor], dim=2)
            new_v = torch.cat([self._cached_kv[1], v_tensor], dim=2)
            self._cached_kv = (new_k, new_v)
        self._offset += k_tensor.size(2)

    def get_cached_kv(self):
        return self._cached_kv
