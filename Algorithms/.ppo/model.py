import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

#TODO the activation function
def mlp(in_features, out_features, hidden_size, activation_func=nn.Tanh()):
    return nn.Sequential(nn.Linear(in_features, hidden_size, bias=True),
                         activation_func,
                         nn.Linear(hidden_size, hidden_size, bias=True),
                         activation_func,
                         nn.Linear(hidden_size, out_features, bias=True),
                         nn.Identity())

class CategoricalActor(nn.Module):
    """
    Categorial Actor, for discrete control space
    """
    def __init__(self, obs_dim, act_dim, hidden_size, activation_func):
        super(CategoricalActor, self).__init__()
        self.logits = mlp(obs_dim, act_dim, hidden_size, activation_func)
    
    def get_distribution(self, obs):
        """
        Categorical object will sample the action with the logits as probability
        """
        logits = self.logits(obs)
        
        return Categorical(logits=logits)
    
    def forward(self, obs, action):
        """
        log_p_a is the probability of sample current action with the distribution
        """
        policy = self.get_distribution(obs)
        log_p_a = policy.log_prob(action)
        
        return policy, log_p_a
    
class Critic(nn.Module):
    """
    value function
    """
    def __init__(self, obs_dim, hidden_size, activation_func):
        super(Critic, self).__init__()
        # output is a value
        self.value_net = mlp(obs_dim, 1, hidden_size, activation_func)
        
    def forward(self, obs):
        return self.value_net(obs)
    
    
class ActorCritic(nn.Module):
    
    def __init__(self, obs_dim, act_dim, hidden_size, activation_func):
        """
        current observation provide 
        critic generate the value function
        
        """
        super(ActorCritic, self).__init__()
        self.actor = CategoricalActor(obs_dim, act_dim, hidden_size, activation_func)
        self.critic = Critic(obs_dim, hidden_size, activation_func)
        
    def step(self, obs):
        with torch.no_grad():
            policy = self.actor.get_distribution(obs)
            action = policy.sample()
            _, log_p_a = self.actor(obs, action)
            value = self.critic(obs)
            
        return action.numpy(), log_p_a.numpy(), value.numpy()
    
    def act(self, obs):
        """
        only return the action
        """
        return self.next_step(obs)[0]