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

    def __init__(self, dim=0, pad_value=0, PTM='GPT2'):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim
        self.pad_value = pad_value
        self.PTM = PTM
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

    def pad_collate_seq2seq(self, batch):
        """
        args:
            batch - list of [encoder_in, label]

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
        """
        # to tensor
        batch = list(map(lambda x: [torch.tensor(x[0]), torch.tensor(x[1])], batch))
        # find longest sequence
        max_len = max(map(lambda x: max(x[0].shape[self.dim],x[1].shape[self.dim]), batch))

        # pad according to max_len
        batch = list(map(lambda x:
                    torch.stack([pad_tensor(x[0], pad=max_len, dim=self.dim, pad_value=self.pad_value),pad_tensor(x[1], pad=max_len, dim=self.dim, pad_value=self.pad_value)], dim=0), batch))
        # stack all
        xs = torch.stack(batch, dim=self.dim)
        return xs

    def pad_collate_xlnet(self, batch):
        """
        args:
            batch - list of [encoder_in, label]

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
        """
        # to tensor
        inputs = list(map(lambda x: torch.tensor(x[0]), batch))
        # find longest sequence
        max_len = max(map(lambda x: x.shape[self.dim], inputs))

        # pad according to max_len
        inputs = list(map(lambda x:
                    pad_tensor(x, pad=max_len, dim=self.dim, pad_value=self.pad_value), inputs))
        # stack all
        xs = [torch.stack(inputs, dim=self.dim), torch.stack([x[1] for x in batch], dim=self.dim) if self.PTM == 'xlnet' else torch.tensor([x[1] for x in batch], dtype=torch.float) ]
        return xs

    def __call__(self, batch):
        if self.PTM == 'GPT2' :
            return self.pad_collate(batch)
        elif self.PTM == 'BART' :
            return self.pad_collate_seq2seq(batch)
        elif self.PTM == 'xlnet' or self.PTM == 'bert' : # bert is for compress usage, training bert model will not use this
            return self.pad_collate_xlnet(batch)

