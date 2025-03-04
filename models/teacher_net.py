import torch


class build_teacher_net(torch.nn.Module):
    def __init__(self, in_channels, stages=3, teacher_dim=1000):
        super(build_teacher_net, self).__init__()
        self.teacher = Teacher(in_channels, stages, teacher_dim)

    def forward(self, x, mask=None):
        if mask is not None:
            masked_x = x * (1 - mask)
        else:
            masked_x = x
        t_out_list = self.teacher(masked_x)

        return t_out_list


class Teacher(torch.nn.Module):
    def __init__(self, in_channels, stages=3, hidden_dim=1000):
        super(Teacher, self).__init__()
        self.layers = torch.nn.ModuleList([])
        for _ in range(stages):
            self.layers.append(TransformerBlock(in_channels, hidden_dim=hidden_dim))

    def forward(self, x):
        # x: [B, C, M, N]
        x = torch.permute(x, [0, 2, 3, 1])
        B, M, N, C = x.shape
        x = torch.reshape(x, [B, M * N, C])
        out_list = []
        for layer in self.layers:
            x = layer(x)
            out_list.append(x.reshape([B, M, N, C]).permute([0, 3, 1, 2]))

        return out_list # list: [B, C, M, N]


class TransformerBlock(torch.nn.Module):
    def __init__(self, in_channels, heads=2, hidden_dim=1000, num_layers=1):
        super(TransformerBlock, self).__init__()
        mlp = TransformerEncoderLayer(in_channels, heads, hidden_dim, batch_first=True)
        self.layers = torch.nn.TransformerEncoder(mlp, num_layers)

    def forward(self, x):
        # x: [B, M*N, C]
        y = self.layers(x)

        return y


class TransformerEncoderLayer(torch.nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation = torch.nn.functional.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        # Legacy string support for activation function.
        self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = torch.nn.functional.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask = None, src_key_padding_mask = None):
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True)[0]

        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
