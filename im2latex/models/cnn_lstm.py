import argparse
from typing import Any, Dict, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

# Common type hints
Param2D = Union[int, Tuple[int, int]]

EMB_SIZE = 80
DEC_RNN_H = 512
ADD_POS_FEAT = False
DROPOUT = 0.0
INIT = 1e-2
ENC_OUT_DIM = 512


class ConvBlock(nn.Module):
    """
    Simple 3x3 conv with padding size 1 (to leave the input size unchanged), followed by a ReLU.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: Param2D = 3,
        stride: Param2D = 1,
        padding: Param2D = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            of dimensions (B, C, H, W)

        Returns
        -------
        torch.Tensor
            of dimensions (B, C, H, W)
        """
        c = self.conv(x)
        r = self.relu(c)
        return r


class CNNLSTM(nn.Module):
    def __init__(
        self, data_config: Dict[str, Any], args: argparse.Namespace = None,
    ):
        super().__init__()
        self.data_config = data_config
        self.num_classes = len(data_config["mapping"])
        inverse_mapping = {val: ind for ind, val in enumerate(data_config["mapping"])}
        self.start_token = inverse_mapping["<S>"]
        self.end_token = inverse_mapping["<E>"]
        self.padding_token = inverse_mapping["<P>"]
        self.max_output_length = data_config["output_dims"][0]
        self.args = vars(args) if args is not None else {}

        out_size = self.num_classes
        enc_out_dim = self.args.get("enc_out_dim", ENC_OUT_DIM)
        emb_size = self.args.get("emb_dim", EMB_SIZE)
        dec_rnn_h = self.args.get("dec_rnn_h", DEC_RNN_H)
        add_pos_feat = self.args.get("add_position_features", ADD_POS_FEAT)
        dropout = float(self.args.get("dropout", DROPOUT))

        self.cnn_encoder = nn.Sequential(
            ConvBlock(1, 64),
            nn.MaxPool2d(2, 2, 1),
            ConvBlock(64, 128),
            nn.MaxPool2d(2, 2, 1),
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d((2, 1), (2, 1), 0),
            ConvBlock(256, enc_out_dim, 3, 1, 0),
        )

        self.rnn_decoder = nn.LSTMCell(dec_rnn_h + emb_size, dec_rnn_h)
        self.embedding = nn.Embedding(out_size, emb_size)

        self.init_wh = nn.Linear(enc_out_dim, dec_rnn_h)
        self.init_wc = nn.Linear(enc_out_dim, dec_rnn_h)
        self.init_wo = nn.Linear(enc_out_dim, dec_rnn_h)

        # Attention mechanism
        self.beta = nn.Parameter(torch.Tensor(enc_out_dim))
        init.uniform_(self.beta, -INIT, INIT)
        self.W_1 = nn.Linear(enc_out_dim, enc_out_dim, bias=False)
        self.W_2 = nn.Linear(dec_rnn_h, enc_out_dim, bias=False)

        self.W_3 = nn.Linear(dec_rnn_h + enc_out_dim, dec_rnn_h, bias=False)
        self.W_out = nn.Linear(dec_rnn_h, out_size, bias=False)

        self.add_pos_feat = add_pos_feat
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, imgs, formulas):
        """args:
        imgs: [B, C, H, W]
        formulas: [B, MAX_LEN]
        epsilon: probability of the current time step to
                 use the true previous token
        return:
        logits: [B, MAX_LEN, VOCAB_SIZE]
        """
        # encoding
        encoded_imgs = self.encode(imgs)  # [B, H*W, 512]
        # init decoder's states
        dec_states, o_t = self.init_decoder(encoded_imgs)
        max_len = formulas.size(1)
        logits = []
        for t in range(max_len):
            tgt = formulas[:, t : t + 1]
            if logits:
                tgt = torch.argmax(torch.log(logits[-1]), dim=1, keepdim=True)
            # ont step decoding
            dec_states, O_t, logit = self.step_decoding(dec_states, o_t, encoded_imgs, tgt)
            logits.append(logit)
        logits = torch.stack(logits, dim=1)  # [B, MAX_LEN, VOCAB_SIZE]
        return logits.permute(0, 2, 1)  # (B, C, Sy)

    def encode(self, imgs: torch.Tensor) -> torch.Tensor:
        encoded_imgs = self.cnn_encoder(imgs)  # [B, 512, H', W']
        encoded_imgs = encoded_imgs.permute(0, 2, 3, 1)  # [B, H', W', 512]
        B, H, W, _ = encoded_imgs.shape
        encoded_imgs = encoded_imgs.contiguous().view(B, H * W, -1)  # [B, H'*W', 512]
        if self.add_pos_feat:
            encoded_imgs = add_positional_features(encoded_imgs)
        return encoded_imgs

    def step_decoding(
        self, dec_states: Tuple[torch.Tensor, torch.Tensor], o_t: torch.Tensor, enc_out: torch.Tensor, tgt: torch.Tensor
    ):
        """Runing one step decoding"""

        prev_y = self.embedding(tgt).squeeze(1)  # [B, emb_size]
        inp = torch.cat([prev_y, o_t], dim=1)  # [B, emb_size+dec_rnn_h]
        h_t, c_t = self.rnn_decoder(inp, dec_states)  # h_t:[B, dec_rnn_h]
        h_t = self.dropout(h_t)
        c_t = self.dropout(c_t)

        # context_t : [B, C]
        context_t, attn_scores = self._get_attn(enc_out, h_t)

        # [B, dec_rnn_h]
        o_t = self.W_3(torch.cat([h_t, context_t], dim=1)).tanh()
        o_t = self.dropout(o_t)

        # calculate logit
        logit = self.W_out(o_t)  # [B, out_size]

        return (h_t, c_t), o_t, logit

    def _get_attn(self, enc_out, h_t):
        """Attention mechanism
        args:
            enc_out: row encoder's output [B, L=H*W, C]
            h_t: the current time step hidden state [B, dec_rnn_h]
        return:
            context: this time step context [B, C]
            attn_scores: Attention scores
        """
        # cal alpha
        alpha = torch.tanh(self.W_1(enc_out) + self.W_2(h_t).unsqueeze(1))
        alpha = torch.sum(self.beta * alpha, dim=-1)  # [B, L]
        alpha = F.softmax(alpha, dim=-1)  # [B, L]

        # cal context: [B, C]
        context = torch.bmm(alpha.unsqueeze(1), enc_out)
        context = context.squeeze(1)
        return context, alpha

    def init_decoder(self, enc_out):
        """args:
            enc_out: the output of row encoder [B, H*W, C]
          return:
            h_0, c_0:  h_0 and c_0's shape: [B, dec_rnn_h]
            init_O : the average of enc_out  [B, dec_rnn_h]
            for decoder
        """
        mean_enc_out = enc_out.mean(dim=1)
        h = self._init_h(mean_enc_out)
        c = self._init_c(mean_enc_out)
        init_o = self._init_o(mean_enc_out)
        return (h, c), init_o

    def _init_h(self, mean_enc_out):
        return torch.tanh(self.init_wh(mean_enc_out))

    def _init_c(self, mean_enc_out):
        return torch.tanh(self.init_wc(mean_enc_out))

    def _init_o(self, mean_enc_out):
        return torch.tanh(self.init_wo(mean_enc_out))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            (B, H, W) image

        Returns
        -------
        torch.Tensor
            (B, Sy) with elements in [0, C-1] where C is num_classes
        """
        B = x.shape[0]
        S = self.max_output_length + 2

        # encoding
        encoded_imgs = self.encode(x)  # [B, H'*W', 512]
        # init decoder's states
        dec_states, O_t = self.init_decoder(encoded_imgs)
        output_tokens = torch.ones(B, S).type_as(x).long() * self.padding_token
        output_tokens[:, 0] = self.start_token  # Set start token
        tgt = torch.ones(B, 1).type_as(x).long() * self.start_token
        # with torch.no_grad():
        for t in range(1, S):
            dec_states, O_t, logit = self.step_decoding(dec_states, O_t, encoded_imgs, tgt)

            tgt = torch.argmax(logit, dim=1, keepdim=True)
            output_tokens[:, t : t + 1] = tgt

        # Set all tokens after end token to be padding
        for Sy in range(1, S):
            ind = (output_tokens[:, Sy - 1] == self.end_token) | (output_tokens[:, Sy - 1] == self.padding_token)
            output_tokens[ind, Sy] = self.padding_token

        return output_tokens  # (B, Sy)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--emb_dim", type=int, default=EMB_SIZE, help="Embedding size")
        parser.add_argument("--dec_rnn_h", type=int, default=DEC_RNN_H, help="The hidden state of the decoder RNN")
        parser.add_argument("--enc_out_dim", type=int, default=ENC_OUT_DIM)
        parser.add_argument(
            "--add_position_features", action="store_true", default=ADD_POS_FEAT, help="Use position embeddings or not"
        )
        parser.add_argument("--dropout", type=float, default=DROPOUT, help="Dropout probability")
        return parser


def add_positional_features(
    tensor: torch.Tensor, min_timescale: float = 1.0, max_timescale: float = 1.0e4
) -> torch.Tensor:
    """
    Implements the frequency-based positional encoding described
    in `Attention is all you Need
    Parameters
    ----------
    tensor : ``torch.Tensor``
        a Tensor with shape (batch_size, timesteps, hidden_dim).
    min_timescale : ``float``, optional (default = 1.0)
        The largest timescale to use.
    Returns
    -------
    The input tensor augmented with the sinusoidal frequencies.
    """
    _, timesteps, hidden_dim = tensor.size()

    timestep_range = get_range_vector(timesteps, tensor.device).data.float()
    # We're generating both cos and sin frequencies,
    # so half for each.
    num_timescales = hidden_dim // 2
    timescale_range = get_range_vector(num_timescales, tensor.device).data.float()

    log_timescale_increments = math.log(float(max_timescale) / float(min_timescale)) / float(num_timescales - 1)
    inverse_timescales = min_timescale * torch.exp(timescale_range * -log_timescale_increments)

    # Broadcasted multiplication - shape (timesteps, num_timescales)
    scaled_time = timestep_range.unsqueeze(1) * inverse_timescales.unsqueeze(0)
    # shape (timesteps, 2 * num_timescales)
    sinusoids = torch.randn(scaled_time.size(0), 2 * scaled_time.size(1), device=tensor.device)
    sinusoids[:, ::2] = torch.sin(scaled_time)
    sinusoids[:, 1::2] = torch.sin(scaled_time)
    if hidden_dim % 2 != 0:
        # if the number of dimensions is odd, the cos and sin
        # timescales had size (hidden_dim - 1) / 2, so we need
        # to add a row of zeros to make up the difference.
        sinusoids = torch.cat([sinusoids, sinusoids.new_zeros(timesteps, 1)], 1)
    return tensor + sinusoids.unsqueeze(0)


def get_range_vector(size: int, device: torch.device) -> torch.Tensor:
    return torch.arange(0, size, dtype=torch.long, device=device)
