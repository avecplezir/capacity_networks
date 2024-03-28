import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        input_channels = min(env.single_observation_space.shape[0], env.single_observation_space.shape[-1])
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 32, 8, stride=4),
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

    def forward(self, x):
        b, ch, w, h = x.shape
        if ch > 32:
            x = x.permute(0, 3, 1, 2).contiguous()
        return self.network(x / 255.0)


class QNetworkMinAtar(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        input_channels = min(env.single_observation_space.shape[0], env.single_observation_space.shape[-1])
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, stride=1),
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

    def forward(self, x):
        b, ch, w, h = x.shape
        if ch > 32:
            x = x.permute(0, 3, 1, 2).contiguous()
        return self.network(x / 255.0)
