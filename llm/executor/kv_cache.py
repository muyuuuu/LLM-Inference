import torch
from llm.model import get_causal_mask


class EasyKVCache:
    def __init__(self):
        self._offset = 0
        self._cached_kv = (None, None)

    def update_and_fetch_kv(self, k_tensor, v_tensor, mask, mask_length=1):
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
        return self._cached_kv[0], self._cached_kv[1], mask, self._offset

    def get_cached_kv(self):
        return self._cached_kv

    def clear_cache(self):
        self._cached_kv = (None, None)
        self._offset = 0


class BatchKVCache:
    def __init__(self, max_activate_requests: int, max_seq_len: int) -> None:
        self._max_activate_requests = max_activate_requests
        self._max_seq_len = max_seq_len
        self._kv_cache = {}

    def update_and_fetch_kv(self, k_tensor, v_tensor, mask, mask_length=1):
        data = []
        batch_size, num_head, seq_len, head_dim = k_tensor.size()
        assert (
            batch_size == self._max_activate_requests
        ), "batch size must be equal to max activate requests"

        max_seq_len = 0
        for i in range(batch_size):
            if i not in self._kv_cache:
                data.append(None)
                continue

            key = k_tensor[i].unsqueeze(0)
            value = v_tensor[i].unsqueeze(0)

            new_key, new_value, mask, sub_seq_len = self._kv_cache[
                i
            ].update_and_fetch_kv(key, value, mask)
            data.append((new_key, new_value, sub_seq_len, mask))
            max_seq_len = max(max_seq_len, sub_seq_len)

        keys = torch.zeros(
            (batch_size, num_head, max_seq_len, head_dim),
            dtype=k_tensor.dtype,
            device=k_tensor.device,
        )
        values = torch.zeros(
            (batch_size, num_head, max_seq_len, head_dim),
            dtype=v_tensor.dtype,
            device=v_tensor.device,
        )
        masks = torch.full(
            (batch_size, mask_length, max_seq_len),
            -float("inf"),
            dtype=k_tensor.dtype,
            device=k_tensor.device,
        )

        for i in range(batch_size):
            if data[i] is None:
                masks[i] = get_causal_mask(mask_length, max_seq_len).to(k_tensor.device)
                continue

            key, value, sub_seq_len, mask = data[i]
            keys[i, :, max_seq_len - sub_seq_len : max_seq_len, :] = key
            values[i, :, max_seq_len - sub_seq_len : max_seq_len, :] = value
            if mask is None:
                masks[i, :, max_seq_len - sub_seq_len : max_seq_len] = get_causal_mask(
                    mask_length, sub_seq_len
                ).to(k_tensor.device)
            else:
                masks[i, :, max_seq_len - sub_seq_len : max_seq_len] = mask.to(
                    k_tensor.device
                )

        return keys, values, masks, max_seq_len

    def add_request(self, prefilled: EasyKVCache, request_idx: int):
        if request_idx >= self._max_activate_requests or request_idx < 0:
            raise ValueError(f"Request {request_idx} out of range")

        assert (
            prefilled.get_cached_kv()[0].size(0) == 1
        ), "prefilled must be for single batch"

        old_request = self._kv_cache.get(request_idx, None)
        if old_request is not None:
            assert old_request.get_cached_kv()[0] is None, "old request must be cleared"

        self._kv_cache[request_idx] = prefilled

    def remove_request(self, request_idx: int):
        if request_idx not in self._kv_cache:
            raise ValueError(f"Request {request_idx} not found")
        self._kv_cache[request_idx].clear_cache()
        del self._kv_cache[request_idx]
