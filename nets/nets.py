import torch
import torch.nn as nn

class QNetworkMinAtar(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 16, 3, stride=1),
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
        return self.network(x)


class QNetworkComposeMinAtar(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.q_networks = nn.ModuleList()
        self.q_networks.append(QNetworkMinAtar(env))

        for _ in range(args.n_q_nets - 1):
            self.q_networks.append(QNetworkMinAtar(env))

    def forward(self, x, return_compose=False):
        x = x.permute(0, 3, 1, 2)
        x = x / 255.0
        q_nets_out = [network(x) for network in self.q_networks]
        q_values = torch.stack(q_nets_out, dim=-1).sum(dim=-1)
        if return_compose:
            return q_values, q_nets_out
        return q_values


class QNetworkMinAtarSimple(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 16, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x)


class QNetworkComposeMinAtarSimple(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.q_networks = nn.ModuleList()
        self.q_networks.append(QNetworkMinAtarSimple(env))

        for _ in range(args.n_q_nets - 1):
            self.q_networks.append(QNetworkMinAtarSimple(env))

    def forward(self, x, return_compose=False):
        x = x.permute(0, 3, 1, 2)
        x = x / 255.0
        q_nets_out = [network(x) for network in self.q_networks]
        q_values = torch.stack(q_nets_out, dim=-1).sum(dim=-1)
        if return_compose:
            return q_values, q_nets_out
        return q_values


class QNetwork(nn.Module):
    def __init__(self, env, args=None):
        super().__init__()
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

    def forward(self, x):
        return self.network(x)

class QNetworkCompose(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.q_networks = nn.ModuleList()
        self.q_networks.append(QNetwork(env))

        for _ in range(args.n_q_nets - 1):
            self.q_networks.append(QNetwork(env))
    def forward(self, x, return_compose=False):
        x = x / 255.0
        q_nets_out = [network(x) for network in self.q_networks]
        q_values = torch.stack(q_nets_out, dim=-1).sum(dim=-1)
        if return_compose:
            return q_values, q_nets_out
        return q_values

class QNetworkEncCapacities(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU())

        self.q_networks = nn.ModuleList()
        self.q_networks.append(nn.Sequential(
           nn.AdaptiveAvgPool2d((1, 1)),
           nn.Flatten(),
           nn.Linear(64, 256),
           nn.ReLU(),
           nn.Linear(256, env.single_action_space.n))
           )

        self.q_networks.append(nn.Sequential(
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n))
            )

    def forward(self, x, return_compose=False):
        x = x / 255.0
        x = self.encoder(x)
        q_nets_out = [network(x) for network in self.q_networks]
        q_values = torch.stack(q_nets_out, dim=-1).sum(dim=-1)
        if return_compose:
            return q_values, q_nets_out
        return q_values


class QNetworkCapacities(QNetworkCompose):
    def __init__(self, env, args):
        nn.Module.__init__(self)
        self.q_networks = nn.ModuleList()
        self.q_networks.append(nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, env.single_action_space.n))
           )

        self.q_networks.append(nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n))
            )

class QNetworkCapacities2(QNetworkCompose):
    def __init__(self, env, args):
        nn.Module.__init__(self)
        self.q_networks = nn.ModuleList()
        self.q_networks.append(nn.Sequential(
            nn.Conv2d(4, 16, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1568, 256),
            nn.ReLU(),
            nn.Linear(256, env.single_action_space.n))
           )

        self.q_networks.append(nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n))
            )

class QNetworkCapacities3(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.q_networks = nn.ModuleList()

        self.q_networks.append(nn.Sequential(
            nn.Conv2d(1, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n))
            )

        self.q_networks.append(nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n))
            )

    def forward(self, x, return_compose=False):
        x = x / 255.0
        q_nets_out = [self.q_networks[0](x[:, -1:]), self.q_networks[1](x)]
        q_values = torch.stack(q_nets_out, dim=-1).sum(dim=-1)
        if return_compose:
            return q_values, q_nets_out
        return q_values


class QNetworkCapacities4(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.q_networks = nn.ModuleList()

        self.q_networks.append(nn.Sequential(
            nn.Conv2d(2, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n))
            )

    def forward(self, x, return_compose=False):
        x = x / 255.0
        q_nets_out = [self.q_networks[0](x[:, -2:]),]
        q_values = torch.stack(q_nets_out, dim=-1).sum(dim=-1)
        if return_compose:
            return q_values, q_nets_out
        return q_values


class QNetworkCapacities5(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.q_networks = nn.ModuleList()

        self.q_networks.append(nn.Sequential(
            nn.Conv2d(2, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n))
            )

    def forward(self, x, return_compose=False):
        x = x / 255.0
        q_nets_out = [self.q_networks[0](x[:, -3:]),]
        q_values = torch.stack(q_nets_out, dim=-1).sum(dim=-1)
        if return_compose:
            return q_values, q_nets_out
        return q_values


class QNetworkCapacities6(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.q_networks = nn.ModuleList()

        self.q_networks.append(nn.Sequential(
            nn.Conv2d(2, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),

            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, env.single_action_space.n))
            )

    def forward(self, x, return_compose=False):
        x = x / 255.0
        q_nets_out = [self.q_networks[0](x[:, -2:]),]
        q_values = torch.stack(q_nets_out, dim=-1).sum(dim=-1)
        if return_compose:
            return q_values, q_nets_out
        return q_values