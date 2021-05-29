import torch

def pad_tensor(vec, pad, dim, pad_value=0):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.full(pad_size, pad_value)], dim=dim)


class PadCollate:
    """
    a variant of collate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0, pad_value=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim
        self.pad_value = pad_value
        # self.max_len = max_len


    def pad_collate(self, batch):
        """
        args:
            batch - list of sequence

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
        """
        # to tensor
        batch = list(map(lambda x: torch.tensor(x), batch))
        # find longest sequence
        max_len = max(map(lambda x: x.shape[self.dim], batch))
        # if max_len >= self.max_len:
        #     batch = list(map(lambda x: x[:self.max_len] if x.shape[0] > self.max_len else x, batch))
        #     max_len = self.max_len
        # pad according to max_len
        batch = list(map(lambda x:
                    pad_tensor(x, pad=max_len, dim=self.dim, pad_value=self.pad_value), batch))
        # stack all
        xs = torch.stack(batch, dim=self.dim)
        return xs

    def __call__(self, batch):
        return self.pad_collate(batch)