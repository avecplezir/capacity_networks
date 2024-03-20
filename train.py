# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

import nets


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "capacity_networks"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 1000000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 1000
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 80000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""
    n_q_nets: int = 1
    """the number of Q networks to compose"""
    reset_transient_frequency: int = -1
    """how often to reset the transient NN to random weights. -1 means never"""
    qnetwork: str = "simple"
    """the type of Q network to use. simple or capacity"""
    pipeline: int = 0
    """the pipeline to use. 0 or 1, 0 is standard cascade value pipeline, 1 -- modification"""

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)

        env.action_space.seed(seed)
        return env

    return thunk

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1" 
"""
        )
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity='avecplezir', #os.getenv('WANDB_USERNAME', 'avecplezir'),
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # if args.qnetwork == "simple":
    #     QNetwork = nets.QNetworkCompose
    # elif args.qnetwork == "capacity":
    #     QNetwork = nets.QNetworkCapacities2
    # elif args.qnetwork == "capacityenc":
    #     QNetwork = nets.QNetworkEncCapacities
    # else:
    #     raise ValueError("unknown agent type")
    QNetwork = getattr(nets, args.qnetwork)
    q_network = QNetwork(envs, args).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs, args).to(device)
    target_network.load_state_dict(q_network.state_dict())
    print('q_network', q_network)

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    targets, q_targets_list = target_network(data.next_observations, return_compose=True)
                    target_actions = torch.argmax(targets, dim=1).unsqueeze(1)
                    v_targets_list = [q.gather(1, target_actions).squeeze() for q in q_targets_list]

                q, q_list = q_network(data.observations, return_compose=True)
                v_list = [q.gather(1, data.actions).squeeze() for q in q_list]

                # r1 + gamma * r2 + gamma^2 * r3
                # r1 - v0 + gamma * (r2 - v1 + v1) + gamma^2 * (r3 - v2 + v2)... ...+ v_0
                # r1 + gamma * v1 - v0  +  gamma * (r2 + gamma*v2 - v1 ) + gamma^2 * (r3 + gamma*v3 - v2)... ...+ v_0
                # r'1 + gamma * r'2 + gamma^2 * r'3
                if args.pipeline == 0:
                    td_targets = []
                    new_reward = data.rewards.flatten()  # r - q1 + gamma * q_target2
                    for i, (v0, v1) in enumerate(zip(v_list, v_targets_list)):
                        td_target = new_reward + args.gamma * v1 * (1 - data.dones.flatten())
                        td_targets.append(td_target.detach())
                        new_reward = td_target - v0
                elif args.pipeline == 1:
                    # cum_sum_q_targets0 = v0+v1+v2+v3+...
                    # cum_sum_q_targets1 = v1+v2+v3+...
                    # ...
                    cum_sum_q_targets = torch.flip(torch.cumsum(torch.stack(q_targets_list[::-1], dim=0), dim=0), dims=[0])
                    cum_sum_q = torch.flip(torch.cumsum(torch.stack(q_list[::-1], dim=0), dim=0), dims=[0])
                    new_reward = data.rewards.flatten()
                    for i, (v0, cum_v_target) in enumerate(zip(v_list, cum_sum_q_targets)):
                        td_target = new_reward + args.gamma * cum_v_target * (1 - data.dones.flatten())
                        td_targets.append(td_target.detach())
                        new_reward = td_target - v0
                else:
                    raise ValueError(f"unknown pipeline {args.pipeline}")

                loss_comp = {}
                if args.pipeline == 0:
                    for i, (q, t) in enumerate(zip(v_list, td_targets)):
                        loss_comp[i] = F.mse_loss(q, t)
                elif args.pipeline == 1:
                    for i, (cq, q, t) in enumerate(zip(cum_sum_q, v_list, td_targets)):
                        loss_comp[i] = F.mse_loss(cq.detach() + q - q.detach(), t)
                else:
                    raise ValueError(f"unknown pipeline {args.pipeline}")

                loss = sum(loss_comp.values())

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    for i, loss in loss_comp.items():
                        writer.add_scalar(f"losses/td_loss_{i}", loss, global_step)
                    for i, v in enumerate(v_list):
                        writer.add_scalar(f"losses/q_values_{i}", v.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

            # reset transient network
            if args.reset_transient_frequency > 0:
                if global_step % args.reset_transient_frequency == 0:
                    q_new_reset = nets.QNetwork(envs).to(device)
                    for target_network_param, q_network_param in zip(q_new_reset.parameters(), q_network.q_networks[0].parameters()):
                        q_network_param.data.copy_(target_network_param.data)
                target_network.q_networks[0].load_state_dict(q_network.q_networks[0].state_dict())


    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.dqn_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=0.05,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "DQN", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()