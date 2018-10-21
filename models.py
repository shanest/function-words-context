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


# TODO: parameterize hidden layers
# TODO: document
class Sender(nn.Module):
    def __init__(self, context_size, n_dims):
        super(Sender, self).__init__()
        self.fc1 = nn.Linear(context_size * n_dims, 32)
        self.fc2 = nn.Linear(32, 32)
        self.dim_msg = nn.Linear(32, n_dims)
        self.min_msg = nn.Linear(32, 2)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        dim_logits = self.dim_msg(x)
        min_logits = self.min_msg(x)
        return F.softmax(dim_logits, dim=1), F.softmax(min_logits, dim=1)


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
        return F.softmax(dim_logits / 0.1, dim=1), F.softmax(min_logits / 0.1, dim=1)


class Receiver(nn.Module):
    def __init__(self, context_size, n_dims):
        super(Receiver, self).__init__()
        self.fc1 = nn.Linear(context_size * n_dims + n_dims + 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 32)
        self.obj_layer = nn.Linear(32, n_dims)
        self.context_size = context_size
        self.n_dims = n_dims

    def forward(self, contexts, msgs):
        x = F.relu(self.fc1(torch.cat([contexts, msgs], dim=1)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        obj = self.obj_layer(x)
        init_comp = (obj.repeat((1, self.context_size)) - contexts)**2
        comp_per_obj = init_comp.view((-1, self.n_dims)).sum(dim=1)
        comp_per_obj = 1 / torch.sqrt(comp_per_obj)
        comp_by_context = comp_per_obj.reshape((-1, self.context_size))
        return obj, F.softmax(comp_by_context / 0.1, dim=1)


# TODO: implement RNN sender and receiver!
