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
            -1e5,
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


class Block:
    def __init__(
        self,
        num_blocks: int,
        num_requests: int,
        num_heads: int,
        head_dim: int,
        block_size=4,
    ) -> None:
        self._num_blocks = num_blocks
        self._num_requests = num_requests
        self._num_heads = num_heads
        self._block_size = block_size
        self._head_dim = head_dim
        self._free = [i for i in range(num_blocks)]
        self._k_cache = None
        self._v_cache = None
        self._request_offset = {i: 0 for i in range(num_requests)}
        try:
            self._k_cache = torch.zeros(num_blocks, num_heads, block_size, head_dim)
            self._v_cache = torch.zeros(num_blocks, num_heads, block_size, head_dim)
        except Exception as e:
            raise ValueError(f"Failed to allocate k_cache and v_cache: {e}")

    def store_kv(self, k_tensor, v_tensor, block_ids, request_idx):
        assert (
            k_tensor.size() == v_tensor.size()
        ), "k_tensor and v_tensor must have the same size"
        seq_len = k_tensor.size(2)
        written = 0
        start = self._request_offset[request_idx]
        while written < seq_len:
            bi = start // self._block_size
            off = start % self._block_size
            n = min(self._block_size - off, seq_len - written)
            self._k_cache[block_ids[bi], :, off : off + n, :] = k_tensor[
                0, :, written : written + n, :
            ].clone()
            self._v_cache[block_ids[bi], :, off : off + n, :] = v_tensor[
                0, :, written : written + n, :
            ].clone()
            written += n
            start += n
        self._request_offset[request_idx] += seq_len

    def should_allocate_new_blocks(self, request_idx):
        assert request_idx in self._request_offset, "request_idx not found"
        if self._request_offset[request_idx] == 0:
            return True
        if self._request_offset[request_idx] % self._block_size == 0:
            return True
        return False

    def gather_kv(self, request_idx, block_ids):
        assert request_idx in self._request_offset, "request_idx not found"
        if self._request_offset[request_idx] == 0:
            return None, None, 0

        out_k = torch.zeros(
            1, self._num_heads, self._request_offset[request_idx], self._head_dim
        )
        out_v = torch.zeros(
            1, self._num_heads, self._request_offset[request_idx], self._head_dim
        )

        start = 0
        for b_idx in block_ids:
            n = min(self._block_size, self._request_offset[request_idx] - start)
            if n <= 0:
                break
            out_k[0, :, start : start + n, :] = self._k_cache[b_idx, :, :n, :].clone()
            out_v[0, :, start : start + n, :] = self._v_cache[b_idx, :, :n, :].clone()
            start += n
        return out_k, out_v, start

    def allocate(self, n: int):
        if len(self._free) < n:
            raise ValueError(f"Need {n} blocks, only {len(self._free)} free")
        return [self._free.pop() for _ in range(n)]

    def free(self, request_idx, block_ids):
        assert request_idx in self._request_offset, "request_idx not found"
        self._request_offset[request_idx] = 0

        for block_id in block_ids:
            assert 0 <= block_id < self._num_blocks, "block_id out of range"
            self._free.append(block_id)


class PagedBatchKVCache:
    def __init__(
        self,
        max_activate_requests: int,
        max_seq_len: int,
        num_blocks: int,
        num_requests: int,
        num_heads: int,
        head_dim: int,
        block_size=4,
        device="cuda",
    ) -> None:
        self._max_activate_requests = max_activate_requests
        self._max_seq_len = max_seq_len
        self._num_blocks = num_blocks
        self._num_requests = num_requests
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._block_size = block_size
        self._device = device
        self._block = Block(num_blocks, num_requests, num_heads, head_dim, block_size)
        self._block_tables = {i: [] for i in range(num_requests)}

    def update_and_fetch_kv(self, k_tensor, v_tensor, mask, mask_length=1):
        batch_size, num_head, seq_len, head_dim = k_tensor.size()
        assert batch_size == self._max_activate_requests

        data = []
        max_seq_len = 0
        for i in range(batch_size):
            if len(self._block_tables[i]) == 0:
                data.append(None)
                continue
            key_i = k_tensor[i].unsqueeze(0)
            value_i = v_tensor[i].unsqueeze(0)

            # allocate blocks for the request
            k_seq_len = key_i.size(2)
            n_blocks = (k_seq_len + self._block_size - 1) // self._block_size
            if self._block.should_allocate_new_blocks(i):
                block_ids = self._block.allocate(n_blocks)
                self._block_tables[i].extend(block_ids)

            self._block.store_kv(key_i, value_i, self._block_tables[i], i)
            new_k, new_v, sub_seq_len = self._block.gather_kv(i, self._block_tables[i])
            data.append((new_k, new_v, sub_seq_len, mask))
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
            -1e5,
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

        kv = prefilled.get_cached_kv()
        k, v = kv[0], kv[1]

        if k is None:
            self._block_tables[request_idx] = []
            return

        assert k.size(0) == 1
        n_blocks = (k.size(2) + self._block_size - 1) // self._block_size
        self._block_tables[request_idx] = self._block.allocate(n_blocks)
        self._block.store_kv(k, v, self._block_tables[request_idx], request_idx)

    def remove_request(self, request_idx: int):
        self._block.free(request_idx, self._block_tables[request_idx])
