import torch
import torch.nn as nn

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
        s, b, ch, w, h = x.shape
        if ch > 32:
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

        self.lstm = nn.LSTM(512, 128, batch_first=False, num_layers=2)
        self.predictor = nn.Linear(128, env.single_action_space.n)

    def forward(self, x, hiddens):
        if len(x.shape) == 4:
            x = x.unsqueeze(0)
        s, b, ch, w, h = x.shape
        if ch > 32:
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


class QNetworkMinAtar(nn.Module):
    def __init__(self, env, args):
        super().__init__()
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