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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: parameterize hidden layers

# TODO: document
class Sender(nn.Module):
    def __init__(self, context_size, n_dims):
        super(Sender, self).__init__()
        self.fc1 = nn.Linear(context_size * n_dims, 32)
        self.fc2 = nn.Linear(32, 16)
        self.dim_msg = nn.Linear(16, n_dims)
        self.min_msg = nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        dim_logits = self.dim_msg(x)
        min_logits = self.min_msg(x)
        return F.softmax(dim_logits, dim=1), F.softmax(min_logits, dim=1)


class Receiver(nn.Module):
    def __init__(self, context_size, n_dims):
        super(Receiver, self).__init__()
        self.fc1 = nn.Linear(context_size * n_dims + 2 + n_dims, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, context_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)


def get_context(n_dims, scale):
    """Gets one 'context'.  A context is a 1-D array representation of
    2*`n_dims` objects.  Each object has `n_dims` properties, where the
    value for each property comes from `scale`.  In a context, each object is
    either the largest or the smallest scalar value along some dimension.

    The context is represented as a shape (4*n_dims) array, where `n_dims`
    elements in a row represent each object.  The order is random.
    """
    objs = []
    for idx in range(n_dims):
        dimension = np.random.choice(scale, size=2*n_dims, replace=False)
        where_min = np.argmin(dimension)
        min_idx = idx * 2
        dimension[[where_min, min_idx]] = dimension[[min_idx, where_min]]
        where_max = np.argmax(dimension)
        max_idx = min_idx + 1
        dimension[[where_max, max_idx]] = dimension[[max_idx, where_max]]
        objs.append(dimension)
    objs = np.array(objs)
    objs = np.transpose(objs)
    np.random.shuffle(objs)
    return np.array(objs).flatten()


def get_permutations(batch_size, context_size):

    return np.stack([np.random.permutation(context_size)
                     for _ in range(batch_size)])


def apply_perms(contexts, perms, n_dims, batch_size):
    # TODO: DOCUMENT
    obj_indices = np.tile(np.stack([np.arange(n_dims)
                                    for _ in range(batch_size)]), 2*n_dims)
    perm_idx = np.repeat(perms, n_dims, axis=1)*n_dims + obj_indices
    return contexts[np.arange(batch_size)[:, None], perm_idx]


if __name__ == '__main__':

    # TODO: argparse stuff!
    n_dims = 3
    objs = np.arange(0, 1, 1/10)
    # TODO: vary context size, not just 2*NDIMS...
    context_size = 2 * n_dims  # number of objects

    batch_size = 16
    num_batches = 50000

    sender = Sender(context_size, n_dims)
    receiver = Receiver(context_size, n_dims)

    sender_opt = torch.optim.Adam(sender.parameters())
    receiver_opt = torch.optim.Adam(receiver.parameters())

    for batch in range(num_batches):

        # 1. get contexts from Nature
        contexts = np.stack([get_context(n_dims, objs)
                            for _ in range(batch_size)])

        # 1a. permute context for receiver
        # NOTE: sender always sends 'first' object in context; receiver sees
        # permuted context
        rec_perms = get_permutations(batch_size, context_size)
        rec_contexts = apply_perms(contexts, rec_perms, n_dims, batch_size)
        # 1b. get correct target based on perms
        target = np.zeros(shape=(batch_size, 1), dtype=np.int64)
        rec_target = rec_perms[np.arange(batch_size)[:, None], target]

        # 2. get signals form sender
        dim_probs, min_probs = sender(torch.Tensor(contexts))
        dim_dist = torch.distributions.OneHotCategorical(dim_probs)
        dim_msg = dim_dist.sample()
        min_dist = torch.distributions.OneHotCategorical(min_probs)
        min_msg = min_dist.sample()

        # 3. get choice from receiver
        choice_probs = receiver(
            torch.cat([torch.Tensor(rec_contexts), dim_msg, min_msg], dim=1))
        choice_dist = torch.distributions.Categorical(choice_probs)
        choice = choice_dist.sample()

        # 4. get reward
        reward = torch.unsqueeze(
            torch.eq(
                torch.from_numpy(rec_target.flatten()),
                choice).float(),
            dim=0)
        # reward 1/0 goes to -1/1
        advantages = 2*reward - 1

        # 5. compute losses and reinforce

        # 5a. sender
        dim_log_prob = dim_dist.log_prob(dim_msg)
        min_log_prob = min_dist.log_prob(min_msg)
        sender_opt.zero_grad()
        sender_loss = -torch.sum(advantages * (dim_log_prob + min_log_prob))
        sender_loss.backward()
        sender_opt.step()

        # 5b. receiver
        choice_log_prob = choice_dist.log_prob(choice)
        receiver_opt.zero_grad()
        receiver_loss = -torch.sum(advantages * choice_log_prob)
        receiver_loss.backward()
        receiver_opt.step()

        print('\nIteration: {}'.format(batch))
        print(contexts)
        print(torch.cat([dim_msg, min_msg], dim=1))
        print(reward)
        print('% correct: {}'.format(torch.mean(reward)))
