import torch
import torch.nn as nn

def loss_pi(actor_critic, data, clip_ratio):
    
    obs, act, advantages, logp_old = data['obs'], data['act'], data['advantages'], data['log_p_a']
    policy, logp_new = actor_critic.actor(obs, act)
    
    # policy improvement
    improvement_ratio = torch.exp(logp_new - logp_old)
    clipped_advantages = torch.clamp(improvement_ratio, 1-clip_ratio, 1+clip_ratio)*advantages
    
    loss = -(torch.min(improvement_ratio * advantages, clipped_advantages).mean())
    kl_div = (logp_old - logp_new).mean().item()
    entropy = policy.entropy().mean().item()
    
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
    loss = loss_func(ret, val)
    
    return loss

def update(actor_critic, data, clip_ratio, train_policy_iters, train_value_iters, actor_optimizer, critic_optimizer, target_kl, writer, epoch):
    
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
        
        