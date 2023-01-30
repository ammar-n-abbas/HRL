##************************** Twin Delayed Deep Deterministic Policy Gradients (TD3) ************************
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e5)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, env.process_history, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, env.process_history, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = np.array(reward)
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


class extract_tensor(nn.Module):
    def forward(self, x):
        tensor, _ = x
        return tensor


class keep_sequence(nn.Module):
    def forward(self, x):
        tensor, _ = x
        return tensor[:, -1, :]


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l0 = nn.LayerNorm(state_dim)
        self.l1 = nn.LSTM(state_dim, 64, batch_first=True)
        self.l2 = nn.LSTM(64, 32, batch_first=True)
        self.l3 = nn.Linear(32, 1)

        self.max_action = max_action

    def forward(self, state):
        a = self.l1(state)[0]
        a = self.l2(a)[0][:, -1, :]
        af = self.l3(a)
        return self.max_action * torch.sigmoid(af)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l0 = nn.LayerNorm(env.process_history * state_dim + action_dim)
        self.l1 = nn.Linear(env.process_history * state_dim + action_dim, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, 1)

        self.l4 = nn.Linear(env.process_history * state_dim + action_dim, 64)
        self.l5 = nn.Linear(64, 32)
        self.l6 = nn.Linear(32, 1)

    def forward(self, state, action):
        sa = torch.cat([torch.reshape(state, (-1, env.process_history * env.feat)), action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([torch.reshape(state, (-1, env.process_history * env.feat)), action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1
