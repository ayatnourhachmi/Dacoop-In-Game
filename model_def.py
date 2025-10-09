import torch
import torch.nn as nn
import numpy as np

class Net(nn.Module):
    """
    Pure inference architecture for the Dueling-like two-head Q network.
    You must pass num_state (length of a single agent observation vector).
    """
    def __init__(self, num_state: int):
        super(Net, self).__init__()
        # Backbone
        self.layer1 = nn.Linear(num_state, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 128)
        self.layer5 = nn.Linear(128, 64)
        self.layer6 = nn.Linear(64, 1)  # state value

        # Heads (discretized APF params)
        self.eta_scale_head = nn.Linear(128, 10)
        self.balance_head = nn.Linear(128, 10)

        # Discrete options identical to training
        self.eta_options = np.linspace(0.1, 10.0, 10)
        self.balance_options = np.linspace(0.0, 4000.0, 10)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))

        adv = torch.relu(self.layer4(x))
        val = torch.relu(self.layer5(x))
        val = self.layer6(val)

        eta_logits = self.eta_scale_head(adv)
        bal_logits = self.balance_head(adv)

        eta_adv = eta_logits - eta_logits.mean(dim=1, keepdim=True)
        bal_adv = bal_logits - bal_logits.mean(dim=1, keepdim=True)

        eta_q = val + eta_adv
        balance_q = val + bal_adv
        return eta_q, balance_q