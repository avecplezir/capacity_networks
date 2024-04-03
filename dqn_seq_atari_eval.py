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


from torch.utils.tensorboard import SummaryWriter

from replay_buffer import ReplayMemory
from nets import nets_seq
from make_envs import make_env

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
    seq_len: int = 1
    """the length of the sequence in the samples"""
    qnetwork: str = "QNetwork"
    """the type of Q network to use"""
    use_relative_attention: bool = False
    """if toggled, use relative attention"""
    policy_regularization_loss: bool = False
    inverse_kl: bool = False
    """if toggled, use inverse KL divergence"""
    collect_eval_data_frequency: int = 1000000


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
    args.device = device

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    args.input_channels = min(envs.single_observation_space.shape[0], envs.single_observation_space.shape[-1])
    args.inverse_channels = envs.single_observation_space.shape[0] > envs.single_observation_space.shape[-1]

    QNetwork = getattr(nets_seq, args.qnetwork)
    q_network = QNetwork(envs, args).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs, args).to(device)
    target_network.load_state_dict(q_network.state_dict())

    net_hiddens = q_network.init_net_hiddens()

    rb = ReplayMemory(
        args.buffer_size,
        envs.single_observation_space.shape,
        action_size=1,
        obs_dtype=envs.single_observation_space.dtype,
        device=device,
        dict=net_hiddens,
    )

    start_time = time.time()
    global_step_threshold = 0

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)

    eval_data = []

    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            with torch.no_grad():
                _, _, next_net_hiddens = q_network(torch.Tensor(obs).to(device), net_hiddens)
        else:
            with torch.no_grad():
                q_values, _, next_net_hiddens = q_network(torch.Tensor(obs).to(device), net_hiddens)
            actions = torch.argmax(q_values[0], dim=1).cpu().numpy()
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    if global_step > global_step_threshold:
                        counts_final_info = 0
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        global_step_threshold += 1000



        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add((obs, real_next_obs, actions, rewards, terminations, infos, net_hiddens))

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        net_hiddens = next_net_hiddens

        if global_step % args.collect_eval_data_frequency == 0:
            eval_data.append(rb.sample_last_seq(args.seq_len))

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample_seq(args.seq_len, args.batch_size)
                with torch.no_grad():
                    # q_value routine
                    target_q_values, targets_q_values_eval, _ = target_network(data.observations, data.net_hiddens)
                    target_max, _ = target_q_values.max(dim=2)
                    td_target = data.rewards[:-1] + args.gamma * target_max[1:] * (1 - data.dones[:-1])

                    # q_value_eval routine
                    target_values_eval = targets_q_values_eval.gather(2, data.actions).squeeze(-1)
                    td_target_eval = data.rewards[:-1] + args.gamma * target_values_eval[1:] * (1 - data.dones[:-1])

                # q_value routine
                old_q_values, q_values_eval, _  = q_network(data.observations, data.net_hiddens)
                old_val = old_q_values.gather(2, data.actions).squeeze(-1)[:-1]
                loss = F.mse_loss(td_target, old_val)

                # q_value_eval routine
                old_eval_val = q_values_eval.gather(2, data.actions).squeeze(-1)[:-1]
                loss_eval = F.mse_loss(td_target_eval, old_eval_val)
                loss += loss_eval

                if args.policy_regularization_loss:
                    def get_dist(x):
                        x_exp = torch.exp(x)
                        sum_exp = torch.sum(x_exp, dim=-1, keepdim=True)
                        return x_exp / sum_exp

                    v_old_proba = get_dist(old_val)
                    v_eval_proba = get_dist(old_eval_val).detach()
                    if args.inverse_kl:
                        reg_loss = torch.mean(torch.sum(v_old_proba * torch.log(v_old_proba / v_eval_proba), dim=-1))
                    else:
                        reg_loss = torch.mean(torch.sum(v_eval_proba * torch.log(v_eval_proba / v_old_proba), dim=-1))
                    loss += reg_loss

                if global_step % 1000 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/td_eval_loss", loss_eval, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    writer.add_scalar("losses/q_values_eval", old_eval_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    if args.policy_regularization_loss:
                        if args.inverse_kl:
                            writer.add_scalar("losses/kl_inv_loss", reg_loss, global_step)
                        else:
                            writer.add_scalar("losses/kl_reg_loss", reg_loss, global_step)

                    for i, data in enumerate(eval_data):
                        with torch.no_grad():
                            q_values, q_values_eval, _ = q_network(data.observations, data.net_hiddens)
                            values, _ = q_values.max(dim=2)
                            values_eval = q_values_eval.gather(2, data.actions).squeeze(-1)
                            writer.add_scalar(f"eval/losses/values_{i}", values.mean().item(), global_step)
                            writer.add_scalar(f"eval/losses/values_eval_{i}", values_eval.mean().item(), global_step)


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