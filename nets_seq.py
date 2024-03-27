import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.args = args
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
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
        x = x.view(s * b, ch, w, h)
        out = self.network(x / 255.0)
        out = out.view(s, b, -1)
        if return_hiddens:
            return out, {}, #{'hidden': None}
        return out

    def init_net_hiddens(self):
        return {}


class QNetworkLSTM(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.args = args
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
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

        self.lstm = nn.LSTM(512, 128, batch_first=False)

        self.predictor = nn.Linear(128, env.single_action_space.n),

    def forward(self, x, hiddens, return_hiddens=False):
        if len(x.shape) == 4:
            x = x.unsqueeze(0)
        s, b, ch, w, h = x.shape
        x = x.view(s * b, ch, w, h)
        out = self.encoder(x / 255.0)
        out = out.view(s, b, -1)

        out, (h, c) = self.lstm(out, (hiddens['h'], hiddens['c']))
        out = self.predictor(out)
        if return_hiddens:
            return out, {'h': h, 'c': c}
        return out

    def init_net_hiddens(self):
        h, c = \
            torch.zeros(self.lstm.num_layers, self.args.num_envs, self.args.lstm.hidden_size).to(self.args.device), \
            torch.zeros(self.lstm.num_layers, self.args.num_envs, self.args.hidden_size).to(self.args.device),
        # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)
        return {'h': h, 'c': c}
