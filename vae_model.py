"""
vae_model.py
============

Shared-attribute variational autoencoder for the joint representation of a
truss design. A design is a coupled pair (M_s, S):

    * M_s : the base-material descriptor (E_s, nu_s), a real-valued vector
            handled by the *material* branch (the ``x`` branch);
    * S   : the structure sentence (a symmetry class followed by its
            connected nodes), a discrete token sequence handled by the
            *structure* branch (the transformer branch).

Both branches are compressed into a partially shared latent space whose
central dimensions encode the coupled material-structure effect. A separate
property predictor maps the latent code to the 21 independent components of
the effective stiffness matrix C (upper triangle of the 6x6 matrix).

This module contains inference-only definitions: model classes, the tokenizer
helpers used to feed structure sentences to the encoder, the greedy decoder
used to read structure sentences back out, and the helpers that turn the
predictor output into C in GPa (with the zeros required by the symmetry
class). No training code is included.
"""

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
device = "cuda:0" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------
# Latent-space bookkeeping
# --------------------------------------------------------------------------
# The shared latent is partitioned as:
#     [ structure-only | shared | material-only ]
# with sizes STRUCT_ONLY, SHARED, MAT_ONLY. The structure branch emits
# STRUCT_ONLY + SHARED dimensions; the material branch emits SHARED + MAT_ONLY;
# the overlapping SHARED block is averaged. Total latent size is
#     D = STRUCT_ONLY + SHARED + MAT_ONLY = 15.

STRUCT_ONLY = 10          # structure-only latent dims  (paper: d_S - d_MS)
SHARED = 3                # shared latent dims          (paper: d_MS)
MAT_ONLY = 2              # material-only latent dims    (paper: d_M - d_MS)
LATENT_DIM = STRUCT_ONLY + SHARED + MAT_ONLY       # = 15

MATERIAL_IN_DIM = 2       # (E_s, nu_s)
MATERIAL_LATENT = SHARED + MAT_ONLY                # material branch latent width
MAX_LEN = 127             # token length of an encoded structure sentence


# --------------------------------------------------------------------------
# Stiffness matrix: names, symmetry zeros, model output -> C in GPa
# --------------------------------------------------------------------------
# The 21 components are the upper triangle of the 6x6 stiffness matrix, row by
# row, in the Voigt order of the data: 1 = 11, 2 = 22, 3 = 33, 4 = 12, 5 = 13,
# 6 = 23 (so C44 is the shear stiffness in the 1-2 plane, C14 couples normal
# stress 1 with shear 12, ...).
UPPER_TRI = [(i, j) for i in range(6) for j in range(i, 6)]
STIFFNESS_NAMES = ["C%d%d" % (i + 1, j + 1) for i, j in UPPER_TRI]
DIAG_COLS = [UPPER_TRI.index((i, i)) for i in range(6)]

# entries that can be non-zero per symmetry class (all others are exactly 0)
_BASE = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2), (3, 3), (4, 4), (5, 5)]
_MONOCLINIC_COUPLING = [(0, 3), (1, 3), (2, 3), (4, 5)]    # C14 C24 C34 C56
_TRIGONAL_COUPLING = [(0, 5), (1, 5), (2, 5), (3, 4)]      # C16 C26 C36 C45
_ALLOWED = {
    "Cubic": _BASE, "Hexagonal": _BASE, "Tetragonal": _BASE, "Orthotropic": _BASE,
    "Monoclinic": _BASE + _MONOCLINIC_COUPLING,
    "Trigonal": _BASE + _TRIGONAL_COUPLING,
}
SYMMETRY_MASK = {cls: np.array([ij in allowed for ij in UPPER_TRI], dtype=float)
                 for cls, allowed in _ALLOWED.items()}


def learned_to_C(y, Es):
    """Predictor values (after scaler_y.inverse_transform) -> C in GPa.

    The predictor learns log10(C / E_s) for the 6 diagonal entries and C / E_s
    for the 15 off-diagonal entries, so C scales exactly with E_s.
    """
    C = np.array(y, dtype=float, copy=True)
    C[:, DIAG_COLS] = 10.0 ** C[:, DIAG_COLS]
    return C * np.asarray(Es, dtype=float).reshape(-1, 1)


def apply_symmetry_zeros(C21, sentences):
    """Set the entries that are zero for the symmetry class (first word of
    each sentence) to exactly 0."""
    mask = np.stack([SYMMETRY_MASK[s.split()[0]] for s in sentences])
    return np.asarray(C21, dtype=float) * mask


def full_matrix(C21):
    """(N, 21) upper triangle -> (N, 6, 6) symmetric stiffness matrices."""
    C21 = np.atleast_2d(np.asarray(C21, dtype=float))
    C = np.zeros((len(C21), 6, 6))
    for k, (i, j) in enumerate(UPPER_TRI):
        C[:, i, j] = C[:, j, i] = C21[:, k]
    return C


def relative_C_error(C21, C21_target):
    """||C - C_target||_F / ||C_target||_F of the full 6x6 matrices."""
    A, B = full_matrix(C21), full_matrix(C21_target)
    return np.linalg.norm(A - B, axis=(1, 2)) / np.linalg.norm(B, axis=(1, 2))


def check_structure(sentence, grammar):
    """Return None if ``sentence`` is a valid structure sentence, otherwise the
    reason. Valid: a known symmetry class followed by distinct node labels of
    that class, with a number of nodes that occurs in the data."""
    words = sentence.split()
    if not words or words[0] not in grammar:
        return "unknown symmetry class; use one of %s" % sorted(grammar)
    labels, n_nodes = grammar[words[0]]["labels"], grammar[words[0]]["n_nodes"]
    body = words[1:]
    bad = [w for w in body if w not in labels]
    if bad:
        return "labels %s are not defined for %s (valid: %s)" % (bad, words[0], labels)
    if len(set(body)) != len(body):
        return "node labels must not repeat"
    if len(body) not in n_nodes:
        return "%s structures have %s nodes, got %d" % (words[0], n_nodes, len(body))
    return None


# --------------------------------------------------------------------------
# Tokenizer helpers for structure sentences
# --------------------------------------------------------------------------
def word_tokenizer(sentence):
    """Split a structure sentence into whitespace-separated tokens."""
    return sentence.split()


def encode_structure(sentences, char_dict, max_len=127):
    """Encode structure sentences into a padded LongTensor of token ids.

    Each sentence is tokenized, mapped through ``char_dict``, framed with a
    leading start id (0), terminated with the ``"<end>"`` id, and padded with
    the ``"_"`` id up to ``max_len``. The trained model expects this exact
    framing (start / tokens / end / pad) -- dropping the end token changes
    what the structure encoder sees and produces a different latent.

    Returns a tensor of shape (n_sentences, max_len).
    """
    pad_id = char_dict["_"]
    end_id = char_dict["<end>"]
    batch = np.full((len(sentences), max_len), pad_id, dtype=np.int64)
    for row, sentence in enumerate(sentences):
        ids = [char_dict[tok] for tok in word_tokenizer(sentence)]
        ids = [0] + ids + [end_id]            # prepend start, append end
        ids = ids[:max_len]
        batch[row, : len(ids)] = ids
    return torch.from_numpy(batch)


def decode_structure(token_batch, org_dict):
    """Turn a batch of decoded token-id rows back into sentences.

    Decoding stops at the first padding or end-of-sequence token of each row.
    """
    sentences = []
    for row in token_batch:
        words = []
        for token_id in row:
            token_id = int(token_id)
            word = org_dict.get(token_id, "_")
            if word in ("_", "<end>"):
                break
            words.append(word)
        sentences.append(" ".join(words))
    return sentences


# --------------------------------------------------------------------------
# Transformer building blocks (structure branch)
# --------------------------------------------------------------------------
def _clone(module, n):
    """Return a ModuleList of ``n`` deep copies of ``module``."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def _subsequent_mask(size):
    """Causal mask that hides not-yet-generated positions."""
    upper = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.uint8)
    return upper == 0


def _attention(query, key, value, mask=None, dropout=None):
    """Scaled dot-product attention."""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weights = scores.softmax(dim=-1)
    if dropout is not None:
        weights = dropout(weights)
    return torch.matmul(weights, value), weights


class MultiHeadedAttention(nn.Module):
    """Multi-head attention with ``h`` heads of width ``d_model // h``."""

    def __init__(self, h, d_model, dropout=0.0):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = _clone(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None, return_attn=False):
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch = query.size(0)
        query, key, value = (
            lin(t).view(batch, -1, self.h, self.d_k).transpose(1, 2)
            for lin, t in zip(self.linears, (query, key, value))
        )
        x, weights = _attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.h * self.d_k)
        if return_attn:
            return weights
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """Two-layer position-wise feed-forward network."""

    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class LayerNorm(nn.Module):
    """Layer normalization."""

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """Residual connection followed by layer normalization."""

    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class Embeddings(nn.Module):
    """Token embedding scaled by sqrt(d_model)."""

    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ConvBottleneck(nn.Module):
    """Three 1-D conv + max-pool blocks that collapse the encoder memory matrix
    down toward a fixed-width latent. Channel widths follow
    ``out = (in - 64) // 2 + 64`` with a final width of 64; kernels are 9 then
    8, 8. ReLU is applied in the forward pass."""

    def __init__(self, size):
        super().__init__()
        blocks = []
        in_d = size
        for i in range(3):
            out_d = int((in_d - 64) // 2 + 64)
            kernel = 9 if i == 0 else 8
            if i == 2:
                out_d = 64
            blocks.append(nn.Sequential(
                nn.Conv1d(in_d, out_d, kernel), nn.MaxPool1d(2)))
            in_d = out_d
        self.conv_layers = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.conv_layers:
            x = F.relu(block(x))
        return x


class DeconvBottleneck(nn.Module):
    """Three 1-D transposed-conv blocks that expand the latent back into the
    decoder memory matrix. Channel widths follow ``out = (size - in) // 4 + in``
    with a final width of ``size``; strides are 4, 3, 1 and the kernel is 11.
    ReLU is applied in the forward pass."""

    def __init__(self, size):
        super().__init__()
        blocks = []
        in_d = 64
        for i in range(3):
            out_d = (size - in_d) // 4 + in_d
            stride = 4 - i
            if i == 2:
                out_d = size
                stride = 1
            blocks.append(nn.Sequential(
                nn.ConvTranspose1d(in_d, out_d, 11, stride=stride, padding=2)))
            in_d = out_d
        self.deconv_layers = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.deconv_layers:
            x = F.relu(block(x))
        return x


class EncoderLayer(nn.Module):
    """Single transformer encoder layer (self-attention + feed-forward)."""

    def __init__(self, size, src_len, self_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.src_len = src_len
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = _clone(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask, return_attn=False):
        attn = None
        if return_attn:
            attn = self.self_attn(x, x, x, mask, return_attn=True)
        x = self.sublayer[0](x, lambda y: self.self_attn(y, y, y, mask))
        out = self.sublayer[1](x, self.feed_forward)
        return (out, attn) if return_attn else out


class DecoderLayer(nn.Module):
    """Single transformer decoder layer (masked self + source attention)."""

    def __init__(self, size, tgt_len, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.tgt_len = tgt_len
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = _clone(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask, return_attn=False):
        m = memory
        x = self.sublayer[0](x, lambda y: self.self_attn(y, y, y, tgt_mask))
        attn = None
        if return_attn:
            attn = self.src_attn(x, m, m, src_mask, return_attn=True)
        x = self.sublayer[1](x, lambda y: self.src_attn(y, m, m, src_mask))
        out = self.sublayer[2](x, self.feed_forward)
        return (out, attn) if return_attn else out


class StructureEncoder(nn.Module):
    """Transformer encoder for the structure branch, with a conv bottleneck
    that produces the structure latent (mean / log-variance) and a predicted
    sequence length used to rebuild the source mask at inference time."""

    def __init__(self, layer, n, d_latent, eps_scale):
        super().__init__()
        self.layers = _clone(layer, n)
        self.conv_bottleneck = ConvBottleneck(layer.size)
        self.z_means = nn.Linear(576, d_latent)
        self.z_var = nn.Linear(576, d_latent)
        self.norm = LayerNorm(layer.size)
        self.predict_len1 = nn.Linear(d_latent, d_latent * 2)
        self.predict_len2 = nn.Linear(d_latent * 2, 127)
        self.eps_scale = eps_scale

    def predict_mask_length(self, latent):
        """Predict the source sequence length from the structure latent."""
        h = self.predict_len2(self.predict_len1(latent))
        return torch.topk(F.softmax(h, dim=-1), 1)[1]

    @staticmethod
    def _reparameterize(mu, logvar, eps_scale=1.0):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * eps_scale * std

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        mem = self.norm(x).permute(0, 2, 1)
        mem = self.conv_bottleneck(mem).contiguous().view(x.size(0), -1)
        mu, logvar = self.z_means(mem), self.z_var(mem)
        z = self._reparameterize(mu, logvar, self.eps_scale)
        pred_len = self.predict_len2(self.predict_len1(mu))
        return z, mu, logvar, pred_len


class StructureDecoder(nn.Module):
    """Transformer decoder for the structure branch. Expands the structure
    latent back into a sequence and attends over it to emit output tokens."""

    def __init__(self, encoder_layer, decoder_layer, n, d_latent):
        super().__init__()
        self.final_encodes = _clone(encoder_layer, 1)
        self.layers = _clone(decoder_layer, n)
        self.norm = LayerNorm(decoder_layer.size)
        self.size = decoder_layer.size
        self.tgt_len = decoder_layer.tgt_len
        self.linear = nn.Linear(d_latent, 576)
        self.deconv_bottleneck = DeconvBottleneck(decoder_layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        mem = F.relu(self.linear(memory)).view(-1, 64, 9)
        mem = self.deconv_bottleneck(mem).permute(0, 2, 1)
        for final_encode in self.final_encodes:
            mem = final_encode(mem, src_mask)
        mem = self.norm(mem)
        for layer in self.layers:
            x = layer(x, mem, src_mask, tgt_mask)
        return self.norm(x)


class Generator(nn.Module):
    """Projects decoder outputs to token-vocabulary logits (log-softmax)."""

    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


# --------------------------------------------------------------------------
# Shared-attribute VAE
# --------------------------------------------------------------------------
class SharedAttributeVAE(nn.Module):
    """Couples a transformer structure branch and a feed-forward material
    branch into one partially shared latent space."""

    def __init__(self):
        super().__init__()
        # transformer hyperparameters (structure branch)
        n_layers = 3
        d_model = 128
        d_ff = 128
        n_heads = 4
        dropout = 0.0
        src_len = 126
        tgt_len = 125
        vocab_size = 35          # token embedding table size
        gen_vocab_size = 34      # generator output size (matches trained model)
        struct_latent = STRUCT_ONLY + SHARED     # structure branch latent width

        attn = MultiHeadedAttention(n_heads, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        self.structure_attn = attn
        self.structure_ff = ff
        self.structure_position = position

        self.structure_encoder = StructureEncoder(
            EncoderLayer(d_model, src_len, copy.deepcopy(attn),
                         copy.deepcopy(ff), dropout),
            n_layers, struct_latent, eps_scale=1.0)

        self.structure_decoder = StructureDecoder(
            EncoderLayer(d_model, src_len, copy.deepcopy(attn),
                         copy.deepcopy(ff), dropout),
            DecoderLayer(d_model, tgt_len, copy.deepcopy(attn),
                         copy.deepcopy(attn), copy.deepcopy(ff), dropout),
            n_layers, struct_latent)

        self.structure_src_embed = nn.Sequential(
            Embeddings(d_model, vocab_size), copy.deepcopy(position))
        self.structure_tgt_embed = nn.Sequential(
            Embeddings(d_model, vocab_size), copy.deepcopy(position))
        self.structure_generator = Generator(d_model, gen_vocab_size)

        # material branch: (E_s, nu_s) -> material latent, and back
        act = nn.ReLU()
        self.material_encoder = nn.Sequential(
            nn.Linear(MATERIAL_IN_DIM, 512), act,
            nn.Linear(512, 512), act,
            nn.Linear(512, 512), act,
            nn.Linear(512, 512), act,
            nn.Linear(512, MATERIAL_LATENT),
        )
        self.material_fc_mu = nn.Linear(MATERIAL_LATENT, MATERIAL_LATENT)
        self.material_fc_logvar = nn.Linear(MATERIAL_LATENT, MATERIAL_LATENT)

        self.material_decoder = nn.Sequential(
            nn.Linear(MATERIAL_LATENT, 512), act,
            nn.Linear(512, 512), act,
            nn.Linear(512, 512), act,
            nn.Linear(512, 512), act,
            nn.Linear(512, MATERIAL_IN_DIM), act,
        )

    @staticmethod
    def _reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def encoder(self, src, src_mask, material):
        """Encode a structure sentence and its base material into the shared
        latent. Returns (z, mu, logvar, predicted_length)."""
        struct_z, struct_mu, struct_logvar, pred_len = self.structure_encoder(
            self.structure_src_embed(src), src_mask)

        mat_h = self.material_encoder(material)
        mat_mu = self.material_fc_mu(mat_h)
        mat_logvar = self.material_fc_logvar(mat_h)

        struct_logvar = torch.clamp(struct_logvar, min=-2.0, max=4.0)
        mat_logvar = torch.clamp(mat_logvar, min=-2.0, max=4.0)

        # concatenate the two branches, then fuse the overlapping shared block
        mu_cat = torch.cat((struct_mu, mat_mu), dim=-1)
        logvar_cat = torch.cat((struct_logvar, mat_logvar), dim=-1)

        batch = mu_cat.size(0)
        mu = torch.zeros(batch, LATENT_DIM, device=mu_cat.device)
        logvar = torch.zeros(batch, LATENT_DIM, device=mu_cat.device)

        s0, sh = STRUCT_ONLY, SHARED
        # structure-only block
        mu[:, :s0] = mu_cat[:, :s0]
        logvar[:, :s0] = logvar_cat[:, :s0]
        # shared block: average the two overlapping estimates
        mu[:, s0:s0 + sh] = 0.5 * (mu_cat[:, s0:s0 + sh]
                                   + mu_cat[:, s0 + sh:s0 + 2 * sh])
        var_a = torch.exp(logvar_cat[:, s0:s0 + sh])
        var_b = torch.exp(logvar_cat[:, s0 + sh:s0 + 2 * sh])
        logvar[:, s0:s0 + sh] = torch.log(0.5 * (var_a + var_b) + 1e-8)
        # material-only block
        mu[:, s0 + sh:] = mu_cat[:, s0 + 2 * sh:]
        logvar[:, s0 + sh:] = logvar_cat[:, s0 + 2 * sh:]

        logvar = torch.clamp(logvar, min=-2.0, max=4.0)
        z = self._reparameterize(mu, logvar)
        return z, mu, logvar, pred_len

    def decoder(self, z, src_mask, tgt, tgt_mask):
        """Decode a latent code back into structure logits and base material."""
        struct_latent = z[:, :STRUCT_ONLY + SHARED]
        mat_latent = z[:, STRUCT_ONLY:]

        struct_hidden = self.structure_decoder(
            self.structure_tgt_embed(tgt), struct_latent, src_mask, tgt_mask)
        struct_logits = self.structure_generator(struct_hidden)
        material_out = self.material_decoder(mat_latent)
        return struct_logits, material_out


# --------------------------------------------------------------------------
# Property predictor
# --------------------------------------------------------------------------
class PropertyPredictor(nn.Module):
    """Maps the latent code to the 21 stiffness components via a shared trunk
    and one small head per component. Outputs are in [0, 1] (scaler_y space);
    use scaler_y.inverse_transform and ``learned_to_C`` to get C in GPa."""

    def __init__(self, latent_dim=LATENT_DIM, shared_dim=512, n_props=21):
        super().__init__()
        self.n_props = n_props

        self.shared = nn.Sequential(
            nn.Linear(latent_dim, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, shared_dim, bias=False), nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(shared_dim, 256, bias=False), nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 128, bias=False), nn.BatchNorm1d(128), nn.ReLU(),
                nn.Linear(128, 1, bias=False), nn.Sigmoid(),
            )
            for _ in range(n_props)
        ])

    def forward(self, z):
        trunk = self.shared(z)
        return torch.cat([head(trunk) for head in self.heads], dim=1)


# --------------------------------------------------------------------------
# Greedy structure decoding
# --------------------------------------------------------------------------
@torch.no_grad()
def greedy_decode(vae, latent, char_dict, org_dict, max_steps=16):
    """Autoregressively decode structure sentences and the base material from a
    batch of latent codes.

    The source mask comes from the sentence length predicted from the latent;
    the output is cut at the first ``<end>`` (everything after it is padding).
    ``max_steps`` = 16 covers the longest sentence (13 words + ``<end>``).

    Returns
        tokens    : (batch, MAX_LEN) LongTensor in the exact encoder format
                    [<start>, words..., <end>, _, _, ...], so a decoded design
                    can be re-encoded directly;
        sentences : list of decoded structure sentences;
        ended     : (batch,) bool array, True if <end> was produced;
        material  : (batch, MATERIAL_IN_DIM) decoded material (scaler_x space).
    """
    dev = latent.device
    start, end, pad = char_dict["<start>"], char_dict["<end>"], char_dict["_"]
    batch = latent.size(0)

    lengths = vae.structure_encoder.predict_mask_length(
        latent[:, :STRUCT_ONLY + SHARED]).view(-1)
    src_mask = (torch.arange(MAX_LEN, device=dev)[None, :] < lengths[:, None]).unsqueeze(-2)

    decoded = torch.full((batch, 1), start, dtype=torch.long, device=dev)
    material = None
    for _ in range(max_steps):
        tgt_mask = _subsequent_mask(decoded.size(1)).to(dev)
        logits, material = vae.decoder(latent, src_mask, decoded, tgt_mask)
        next_word = logits[:, -1].argmax(-1) + 1     # generator has no <start> class
        decoded = torch.cat([decoded, next_word[:, None]], dim=1)

    words = decoded[:, 1:]
    is_end = words == end
    ended = is_end.any(1)
    first_end = torch.where(ended, is_end.float().argmax(1),
                            torch.full_like(lengths, words.size(1)))
    pos = torch.arange(words.size(1), device=dev)[None, :]
    words = torch.where(pos < first_end[:, None], words, torch.full_like(words, pad))
    words = torch.where(pos == first_end[:, None], torch.full_like(words, end), words)

    tokens = torch.full((batch, MAX_LEN), pad, dtype=torch.long, device=dev)
    tokens[:, 0] = start
    tokens[:, 1:1 + words.size(1)] = words
    sentences = [" ".join(org_dict[int(t)] for t in row[1:1 + int(n)].tolist())
                 for row, n in zip(tokens.cpu(), first_end.cpu())]
    return tokens, sentences, ended.cpu().numpy(), material
