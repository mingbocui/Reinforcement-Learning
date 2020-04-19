import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import gym
import time
import random
from model import ActorCritic
from buffer import Buffer
from loss import update



# buffer will be reset for every epoch
# 因为advantage只能灯rewards和value计算完再计算
# 每个episode有长有短，我们储存一个episode里面每个step的reward和value，等到episode结束后才能逆序计算advantages

# 定义每个trajectory的最长的长度 traj_max_len
# 当前trajectory的steps超过traj_max_len时候会判断是timeout
# 每个trajectory达到terminal的状态有两种情况， 一种是达到了目标状态， 即 done = True; 另一种是timeout
# 如果达到了目标状态，那么在算 additional value的时候应该设为0
# 如果是因为达到epoch内最大的steps（比如每个epoch只能有1000个step）或者因为当前的trajectory达到最大的steps数导致的终止， 我们需要
# bootstrap value
def run_ppo(env, hidden_size, buffer_size, policy_lr, val_lr, epochs, steps_per_epoch, traj_max_len, gamma, lam,
           clip_ratio, train_policy_iters, train_value_iters, target_kl, writer):
    
#     env = env()
    
#     obs_dim = env.observation_space.shape
#     act_dim = env.action_space.shape
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    print("here", obs_dim, act_dim)
    
    
    actor_critic = ActorCritic(obs_dim, act_dim, hidden_size, activation_func=nn.Tanh())
    
    actor_optimizer = Adam(actor_critic.actor.parameters(), lr=policy_lr)
    critic_optimizer = Adam(actor_critic.critic.parameters(), lr=val_lr)
    
    print("RUNNING")
    obs = env.reset()
    epoch_ret = 0
    traj_len = 0
    
    buffer = Buffer(obs_dim, act_dim, buffer_size, gamma=0.99, lam=0.95)
    
    for epoch in range(epochs):
        # ppo is online learning so every epoch we need to recreat a buffer
        buffer.reset()
        for step in range(steps_per_epoch):
            # 根据当前的observation通过policy net来生成logits，logits作为probability distribution 用来sample选择action
            # 注意这里的obs是一维度的
            act, logp, val = actor_critic.step(torch.as_tensor(obs, dtype=torch.float32))
#             action.numpy(), log_p_a.numpy(), value.numpy()
#             print("act", act.shape)
            next_obs, reward, done, _ = env.step(act)
            epoch_ret += reward
            traj_len += 1
            buffer.store(obs, act, reward, val, logp)
            
            # 需要下一步的observation去get bootstap if the trajectory ended "unnormally"
            obs = next_obs
            
            timeout = traj_len == traj_max_len
            terminal = done or timeout
            epoch_end = step == steps_per_epoch-1
            
            # spinnup的代码似乎考虑了多线程运行
            # 因此它们的 buffer size是steps_per_epoch / num_procs
            # 即每个procs有一个buffer
            
            # 只需要在terminal或者epoch_end的情况下需要运行
            if terminal or epoch_end:
                #当trajectory“非正常”终止的时候才需要bootstrap
                if epoch_end or timeout:
                    _, val, _ = actor_critic.step(torch.as_tensor(o, dtype=torch.float32))
                    
                #如果正常终止，因为达到了目标地点从而 done = True
                #这种情况下不应该得到任何的additional value
                else:
                    val = 0
                
                buffer.finish_with(val)
                o, ep_ret, traj_len = env.reset(), 0, 0
        data = buffer.get_buf_data()
        update(actor_critic, data, clip_ratio, train_policy_iters, train_value_iters, actor_optimizer, critic_optimizer, target_kl, writer, epoch)
        print("Epoch ended")

if __name__ == '__main__':
    
    from tensorboardX import SummaryWriter
    import time
    log_dir = "logs"
    post_fix = '/' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_dir = log_dir + post_fix

    writer = SummaryWriter(log_dir=log_dir)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='LunarLander-v2') #HalfCheetah-v2, CartPole-v0, MountainCar-v0 #LunarLander-v2
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.97)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--steps_per_epoch', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--policy_lr', type=float, default=3e-4)
    parser.add_argument('--val_lr', type=float, default=1e-3)
    parser.add_argument('--train_policy_iters', type=int, default=80)
    parser.add_argument('--train_value_iters', type=int, default=80)
    parser.add_argument('--traj_max_len', type=int, default=1000)
    parser.add_argument('--clip_ratio', type=float, default=0.2)
    parser.add_argument('--target_kl', type=float, default=0.01)
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
#     env = gym.make(args.env)
    
    run_ppo(env=gym.make(args.env), 
            hidden_size=args.hidden_size, 
            buffer_size=args.steps_per_epoch,
            policy_lr=args.policy_lr, 
            val_lr=args.val_lr,
            epochs=args.epochs, 
            steps_per_epoch=args.steps_per_epoch, 
            traj_max_len=args.traj_max_len, 
            gamma=args.gamma, 
            lam=args.lam,
            clip_ratio=args.clip_ratio, 
            train_policy_iters=args.train_policy_iters,
            train_value_iters=args.train_value_iters, 
            target_kl=args.target_kl,
            writer=writer)
    
    