import torch
import torch.nn as nn

def loss_pi(actor_critic, data, clip_ratio):
    obs, act, advantages, logp_old = data['obs'], data['act'], data['advantages'], data['log_p_a']
    # policy net will observe the observations and actions collected in the corrent epoch
    # the logp_old is log(pi), which is the part of the policy gradient
    
    # given observation, we use the MLP to regress the logits, use the logits as sampled probability
    # return policy is the probability distribution under sertain observation
    # logp_new is the log probability when chosen the action
    # policy是包含了4000个observation的probability distribution， 如果action是5维度那么它的维度是 4000 x 5
    # logp_new相当于slice，当前4000个action对应的log probability所以维度是4000 x 1
    # 这里的policy是新的policy，因为在上一次policy net被update过weights
#     print("obs", obs.shape, act.shape, advantages.shape, logp_old.shape)
    policy, logp_new = actor_critic.actor(obs, act)
    
    
    # policy improvement
    improvement_ratio = torch.exp(logp_new - logp_old)
    clipped_advantages = torch.clamp(improvement_ratio, 1-clip_ratio, 1+clip_ratio)*advantages
    
    # 如果当前的update ratio没有超过1-ratio, 1+ratio的这个范围，就用真实的advantages来policy gradient
    # 因为是gain reward, 所以用负号来用
    loss = -(torch.min(improvement_ratio * advantages, clipped_advantages).mean())
    
    # KL divergence是log(P/Q), 分解出来是log(P) - log(Q)
    kl_div = (logp_old - logp_new).mean().item()
    
    # pi 是 categorical 类，自带。entropy()方法， 计算的其实就是 np.sum(-pi*log(pi))
    entropy = policy.entropy().mean().item()
    
    #计算policy(4000x5)这个矩阵中超过1+ratio和小于1-ratio的范围
    clipped = improvement_ratio.gt(1+clip_ratio) | improvement_ratio.lt(1-clip_ratio)
    
    clipped_frac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
    
    policy_info = dict(kl_div=kl_div, entropy=entropy, clipped_frac=clipped_frac)
    
    return loss, policy_info
    
    
    
def loss_val(actor_critic, data):
    obs, ret = data['obs'], data['ret']
    
    # given observation to predict the val
    # discounted return is the target we would like to regress to
    val = actor_critic.critic(obs)
    
    loss_func = nn.MSELoss(reduction='mean')
    
    val = torch.squeeze(val, 1)
#     print("ret, val", ret.shape, val.shape)
    
    loss = loss_func(ret, val)
    
    return loss

def update(actor_critic, data, clip_ratio, train_policy_iters, train_value_iters, actor_optimizer, critic_optimizer, target_kl, writer, epoch):
    
#     policy_loss_old, policy_info_old = loss_pi(ac, data, clip_ratio)
#     policy_loss_old = policy_loss_old.item()
#     val_loss_old = loss_val(ac, data).item()
    
    for i in range(train_policy_iters):
        actor_optimizer.zero_grad()
        policy_loss, policy_info = loss_pi(actor_critic, data, clip_ratio)
        kl_div = policy_info['kl_div']
        if kl_div > 1.5 * target_kl:
            print("Reaching max kl, break at epoch {}".format(epoch))
            break
        policy_loss.backward()
        actor_optimizer.step()
        
    for i in range(train_value_iters):
        critic_optimizer.zero_grad()
        value_loss = loss_val(actor_critic, data)
        value_loss.backward()
        critic_optimizer.step()
        
    writer.add_scalars('scalar', {'policy_loss': policy_loss.item(), 
                                  'value_loss': value_loss.item()}, epoch)
        
    # TODO use logger to log changes from update
        
        