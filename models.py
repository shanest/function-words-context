"""
Copyright (C) 2018 Shane Steinert-Threlkeld
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: parameterize hidden layers, softmax temps
# TODO: document
class Sender(nn.Module):
    def __init__(self, context_size, n_dims, with_dim_labels):
        super(Sender, self).__init__()
        self.fc1 = nn.Linear(context_size * n_dims**(1+int(with_dim_labels)), 32)
        self.fc2 = nn.Linear(32, 32)
        self.dim_msg = nn.Linear(32, n_dims)
        self.min_msg = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        dim_logits = self.dim_msg(x)
        min_logits = self.min_msg(x)
        # TODO: refactor these 4 lines into a util method?
        msg_probs = (F.softmax(dim_logits / 5, dim=1),
                     F.softmax(min_logits / 5, dim=1))
        msg_dists = [torch.distributions.OneHotCategorical(probs)
                     for probs in msg_probs]
        msgs = [dist.sample() for dist in msg_dists]
        return msg_dists, msgs


# TODO: make all agents compatible with with_dim_labels

class SplitSender(nn.Module):
    def __init__(self, context_size, n_dims):
        super(SplitSender, self).__init__()
        self.dim1 = nn.Linear(context_size * n_dims, 32)
        self.dim2 = nn.Linear(32, 32)
        self.min1 = nn.Linear(context_size * n_dims, 32)
        self.min2 = nn.Linear(32, 32)
        self.dim_msg = nn.Linear(32, n_dims)
        self.min_msg = nn.Linear(32, 2)

    def forward(self, x):
        dimx = F.relu(self.dim1(x))
        dimx = F.relu(self.dim2(dimx))
        dim_logits = self.dim_msg(dimx)
        minx = F.relu(self.min1(x))
        minx = F.relu(self.min2(minx))
        min_logits = self.min_msg(minx)
        msg_probs = (F.softmax(dim_logits / 1, dim=1),
                     F.softmax(min_logits / 1, dim=1))
        msg_dists = [torch.distributions.OneHotCategorical(probs)
                     for probs in msg_probs]
        msgs = [dist.sample() for dist in msg_dists]
        return msg_dists, msgs


class RNNSender(nn.Module):

    def __init__(self, context_size, n_dims, with_dim_labels,
                 max_len=4, num_msgs=2, hidden_size=64):
        super(RNNSender, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTMCell(context_size * n_dims + num_msgs + hidden_size,
                                hidden_size)
        self.msg = nn.Linear(hidden_size, num_msgs)
        self.max_len = max_len
        self.num_msgs = num_msgs

    def forward(self, contexts):
        batch_size = contexts.shape[0]
        hidden = torch.zeros((batch_size, self.hidden_size))
        cur_msg = torch.zeros((batch_size, self.num_msgs))
        msg_dists, msgs = [], []
        for _ in range(self.max_len):
            combined = torch.cat([contexts, cur_msg, hidden], dim=1)
            hidden, output = self.lstm(combined)
            msg_probs = F.softmax(self.msg(output) / 0.25, dim=1)
            msg_dists.append(torch.distributions.OneHotCategorical(msg_probs))
            cur_msg = msg_dists[-1].sample()
            msgs.append(cur_msg)
        return msg_dists, msgs


class BaseReceiver(nn.Module):
    def __init__(self, context_size, n_dims, max_msg, with_dim_labels):
        super(BaseReceiver, self).__init__()
        self.fc1 = nn.Linear(context_size * n_dims**(1+int(with_dim_labels))
                             + n_dims + 2,  64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 32)
        self.target = nn.Linear(32, context_size)

    def forward(self, contexts, msgs):
        x = F.relu(self.fc1(torch.cat([contexts] + msgs, dim=1)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        target = self.target(x)
        return F.softmax(target / 5, dim=1)


class DimReceiver(nn.Module):
    # TODO: unify interface with this agent and others
    def __init__(self, context_size, n_dims, max_msg, with_dim_labels):
        super(DimReceiver, self).__init__()
        self.fc1 = nn.Linear(context_size * n_dims + n_dims + 2 +
                             int(with_dim_labels)*context_size*n_dims**2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 32)
        self.target = nn.Linear(32, context_size)
        self.dim = nn.Linear(n_dims+2, 16)
        self.dim2 = nn.Linear(16, n_dims)

    def forward(self, contexts, msgs):
        x = F.relu(self.fc1(torch.cat([contexts] + msgs, dim=1)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        target = self.target(x)
        dim = F.relu(self.dim(torch.cat(msgs, dim=1)))
        dim = self.dim2(dim)
        return (F.softmax(target / 0.5, dim=1),
                F.softmax(dim / 0.5, dim=1))


class MSEReceiver(nn.Module):
    def __init__(self, context_size, n_dims, max_msg, with_dim_labels):
        super(MSEReceiver, self).__init__()
        self.fc1 = nn.Linear(context_size * n_dims + n_dims + 2 +
                             int(with_dim_labels)*context_size*n_dims**2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 32)
        self.obj_layer = nn.Linear(32, n_dims + int(with_dim_labels)*n_dims**2)
        self.context_size = context_size
        self.n_dims = n_dims
        self.with_dim_labels = with_dim_labels

    def forward(self, contexts, msgs):
        x = F.relu(self.fc1(torch.cat([contexts] + msgs, dim=1)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        obj = self.obj_layer(x)
        # TODO: parameterize this receiver based on similarity measure here?
        init_comp = (obj.repeat((1, self.context_size)) - contexts)**2
        comp_per_obj = init_comp.view(
            (-1, self.n_dims + int(self.with_dim_labels)*self.n_dims**2)
        ).sum(dim=1)
        comp_per_obj = 1 / torch.sqrt(comp_per_obj)
        target = comp_per_obj.reshape((-1, self.context_size))
        return F.softmax(target / 2, dim=1)


class RNNReceiver(nn.Module):

    def __init__(self, context_size, n_dims, max_msg, with_dim_labels,
                 hidden_size=64):
        super(RNNReceiver, self).__init__()
        self.hidden_size = hidden_size
        self.max_msg = max_msg
        self.lstm = nn.LSTMCell(context_size * n_dims + max_msg + hidden_size,
                                hidden_size)
        self.target = nn.Linear(hidden_size, context_size)

    def forward(self, contexts, msgs):
        hidden = torch.zeros(contexts.shape[0], self.hidden_size)
        for msg in msgs:
            num_msg = msg.shape[1]
            if num_msg < self.max_msg:  # can have diff msg len per pos
                msg = torch.cat(
                    [msg, torch.zeros((msg.shape[0], self.max_msg - num_msg))],
                    dim=1)
            combined = torch.cat([contexts, msg, hidden], dim=1)
            hidden, output = self.lstm(combined)
        target = self.target(output)
        return F.softmax(target / 0.25, dim=1)
