import torch
import torch.nn as nn

from collections import namedtuple

class QNetwork(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.args = args
        self.network = nn.Sequential(
            nn.Conv2d(args.input_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n),
        )

    def forward(self, x, hiddens, return_hiddens=False):
        if len(x.shape) == 4:
            x = x.unsqueeze(0)
        if self.args.inverse_channels:
            x = x.permute(0, 1, -1, -3, -2).contiguous()
        s, b, ch, w, h = x.shape
        x = x.view(s * b, ch, w, h)
        out = self.network(x / 255.0)
        out = out.view(s, b, -1)
        if return_hiddens:
            return out, {}
        return out

    def init_net_hiddens(self):
        return {}


class QNetworkLSTM(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.args = args
        self.encoder = nn.Sequential(
            nn.Conv2d(args.input_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )

        self.lstm = nn.LSTM(512, 128, batch_first=False, num_layers=1)
        self.predictor = nn.Linear(128, env.single_action_space.n)

    def forward(self, x, hiddens):
        if len(x.shape) == 4:
            x = x.unsqueeze(0)
        if self.args.inverse_channels:
            x = x.permute(0, 1, -1, -3, -2).contiguous()
        s, b, ch, w, h = x.shape
        x = x.view(s * b, ch, w, h)
        out = self.encoder(x / 255.0)
        out = out.view(s, b, -1)
        if len(hiddens['h'].shape) == 4:
            (h, c) = (hiddens['h'][0], hiddens['c'][0])
        else:
            (h, c) = hiddens['h'], hiddens['c']
        (h, c) = h.transpose(0, 1).contiguous(), c.transpose(0, 1).contiguous()
        out, (h, c) = self.lstm(out, (h, c))
        (h, c) = h.transpose(0, 1), c.transpose(0, 1)
        out = self.predictor(out)
        return out, {'h': h, 'c': c}

    def init_net_hiddens(self):
        h, c = \
            torch.zeros(self.args.num_envs, self.lstm.num_layers, self.lstm.hidden_size).to(self.args.device), \
            torch.zeros(self.args.num_envs, self.lstm.num_layers, self.lstm.hidden_size).to(self.args.device),
        # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)
        return {'h': h, 'c': c}

class QNetworkLSTMNamedTuple(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.args = args
        self.q_value_dim = env.single_action_space.n
        self.encoder = nn.Sequential(
            nn.Conv2d(args.input_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )

        self.lstm = nn.LSTM(512, 128, batch_first=False, num_layers=2)
        self.predictor = nn.Linear(128, self.q_value_dim)
        self.ActorOut = namedtuple(
            "ActorOut",
            [
                "q_values",
                "h",
                "c",
            ],
        )

    def forward(self, data):
        x = data.observation
        if len(x.shape) == 4:
            x = x.unsqueeze(0)
        if self.args.inverse_channels:
            x = x.permute(0, 1, -1, -3, -2).contiguous()
        s, b, ch, w, h = x.shape
        x = x.view(s * b, ch, w, h)
        out = self.encoder(x / 255.0)
        out = out.view(s, b, -1)
        if len(data.h.shape) == 4:
            (h, c) = (data.h[0], data.c[0])
        else:
            (h, c) = data.h, data.c
        (h, c) = h.transpose(0, 1).contiguous(), c.transpose(0, 1).contiguous()
        out, (h, c) = self.lstm(out, (h, c))
        (h, c) = h.transpose(0, 1), c.transpose(0, 1)
        q_values = self.predictor(out)
        out = self.ActorOut(q_values=q_values, h=h, c=c)
        return out

    def init_net_hiddens(self):
        q_values = -torch.ones(self.args.num_envs, self.q_value_dim).to(self.args.device)
        h, c = \
            torch.zeros(self.args.num_envs, self.lstm.num_layers, self.lstm.hidden_size).to(self.args.device), \
            torch.zeros(self.args.num_envs, self.lstm.num_layers, self.lstm.hidden_size).to(self.args.device),
        return self.ActorOut(q_values=q_values, h=h, c=c)

class QNetworkMinAtar(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.args = args
        self.network = nn.Sequential(
            nn.Conv2d(args.input_channels, 16, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, env.single_action_space.n),
        )

    def forward(self, x, hiddens):
        if len(x.shape) == 4:
            x = x.unsqueeze(0)
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        s, b, ch, w, h = x.shape
        x = x.view(s * b, ch, w, h)
        out = self.network(x / 255.0)
        out = out.view(s, b, -1)
        return out, {}

    def init_net_hiddens(self):
        return {}


class QNetworkMinAtarLSTM(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.args = args
        self.encoder = nn.Sequential(
            nn.Conv2d(args.input_channels, 16, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128, 256),
        )

        self.lstm = nn.LSTM(256, 128, batch_first=False, num_layers=1)
        self.predictor = nn.Linear(128, env.single_action_space.n)

    def forward(self, x, hiddens):
        if len(x.shape) == 4:
            x = x.unsqueeze(0)
        # inverse channels for minatari
        x = x.permute(0, 1, -1, -3, -2).contiguous()
        s, b, ch, w, h = x.shape
        x = x.view(s * b, ch, w, h)
        out = self.encoder(x / 255.0)
        out = out.view(s, b, -1)
        if len(hiddens['h'].shape) == 4:
            (h, c) = (hiddens['h'][0], hiddens['c'][0])
        else:
            (h, c) = hiddens['h'], hiddens['c']
        (h, c) = h.transpose(0, 1).contiguous(), c.transpose(0, 1).contiguous()
        out, (h, c) = self.lstm(out, (h, c))
        (h, c) = h.transpose(0, 1), c.transpose(0, 1)
        out = self.predictor(out)
        return out, {'h': h, 'c': c}

    def init_net_hiddens(self):
        h, c = \
            torch.zeros(self.args.num_envs, self.lstm.num_layers, self.lstm.hidden_size).to(self.args.device), \
            torch.zeros(self.args.num_envs, self.lstm.num_layers, self.lstm.hidden_size).to(self.args.device),
        # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)
        return {'h': h, 'c': c}


from nets.attention import AttentionBlock, PositionalEncoding, RelativeAttentionBlock
from collections import deque

class QNetworkMinAtarTransformer(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.args = args
        self.encoder = nn.Sequential(
            nn.Conv2d(args.input_channels, 16, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128, 256),
        )

        if args.use_relative_attention:
            self.attn = RelativeAttentionBlock(n_embd=256, n_head=8, block_size=args.seq_len)
        else:
            self.pos_encoder = PositionalEncoding(256)
            self.attn = AttentionBlock(n_embd=256, n_head=8, block_size=args.seq_len)
        self.predictor = nn.Linear(256, env.single_action_space.n)

        self.online_previous_enc = deque(maxlen=args.seq_len)

    def forward(self, x, hiddens):
        collect_prev_enc = (len(x.shape) == 4)
        if len(x.shape) == 4:
            x = x.unsqueeze(0)

        # inverse channels for minatari
        x = x.permute(0, 1, -1, -3, -2).contiguous()
        s, b, ch, w, h = x.shape
        x = x.view(s * b, ch, w, h)
        out = self.encoder(x / 255.0)
        out = out.view(s, b, -1)
        if collect_prev_enc:
            self.online_previous_enc.append(out)
            out = torch.concat(list(self.online_previous_enc), dim=0)
        if not self.args.use_relative_attention:
            out = self.pos_encoder(out)
        out = out.permute(1, 0, 2).contiguous()
        out = self.attn(out)
        out = out.permute(1, 0, 2).contiguous()
        out = self.predictor(out)
        if collect_prev_enc:
            out = out[-1:]
        return out, {}

    def init_net_hiddens(self):
        return {}