import torch


def get_seq_len(data):
    if data is None:
        return 0
    _, _, seq_len, _ = data
    return seq_len


data = []
data.append(None)
data.append((1, 2, 3, 4))
data.append((1, 2, 5, 6))

seq_len = max(map(get_seq_len, data))
print(seq_len)
