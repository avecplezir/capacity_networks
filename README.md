Atari 
===================

DQN is adapted from CleanRL 
(https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py)

Some runs:

MinAtari, deep network, one critic:

```
python train.py --qnetwork QNetworkComposeMinAtar \
                --env_id MinAtar/Breakout-v1 \
                --n_q_nets 1 \
                --exp_name QNetworkComposeMinAtar_1 \
                --track
```

MinAtari, deep network, two cascade critic:

```
python train.py --qnetwork QNetworkComposeMinAtar \
                --env_id MinAtar/Breakout-v1 \
                --n_q_nets 2 \
                --exp_name QNetworkComposeMinAtar_2 \
                --track
```

MinAtari, shallow network, one critic:

```
python train.py --qnetwork QNetworkMinAtarSimple \
                --env_id MinAtar/Breakout-v1 \
                --exp_name nq_1 \
                --n_q_nets 1 \
                --exp_name QNetworkMinAtarSimple_1 \
                --track
```

Atari, two cascade critic:

```
python train.py --qnetwork QNetworkCompose \
                --env_id BreakoutNoFrameskip-v4 \
                --exp_name nq_2_p2 \
                --n_q_nets 2 \
                --track
```