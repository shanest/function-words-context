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
from tensorboardX import SummaryWriter  # TODO: is tbX worth it?


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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        dim_logits = self.dim_msg(x)
        min_logits = self.min_msg(x)
        return F.softmax(dim_logits, dim=1), F.softmax(min_logits, dim=1)


class Receiver(nn.Module):
    def __init__(self, context_size, n_dims):
        super(Receiver, self).__init__()
        self.fc1 = nn.Linear(context_size * n_dims + n_dims + 2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, n_dims)
        # TODO: currently, predicting feature vals; turn back to one-hot?

    def forward(self, contexts, dim_msg, min_msg):
        x = F.relu(self.fc1(torch.cat([contexts, dim_msg, min_msg], dim=1)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        # return F.softmax(x, dim=1)
        return F.sigmoid(x)


class LSTMReceiver(nn.Module):

    def __init__(self, context_size, n_dims):
        super(LSTMReceiver, self).__init__()
        # TODO: update to when n_dims != 2; need to make min_msg and dim_msg
        # have same dimensionality
        self.cell = nn.LSTMCell(context_size * n_dims + n_dims, 32)
        self.fc = nn.Linear(32, context_size)

    def forward(self, contexts, dim_msg, min_msg):
        # no initial h, c = default 0
        hx, cx = self.cell(torch.cat([contexts, dim_msg], dim=1))
        hx, cx = self.cell(torch.cat([contexts, min_msg], dim=1), (hx, cx))
        out = self.fc(hx)
        return F.softmax(out, dim=1)


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
    return objs.flatten()


def get_permutations(batch_size, context_size):

    return np.stack([np.random.permutation(context_size)
                     for _ in range(batch_size)])


def apply_perms(contexts, perms, n_dims, batch_size):
    # TODO: DOCUMENT
    obj_indices = np.tile(np.stack([np.arange(n_dims)
                                    for _ in range(batch_size)]), 2*n_dims)
    perm_idx = np.repeat(perms, n_dims, axis=1)*n_dims + obj_indices
    return contexts[np.arange(batch_size)[:, None], perm_idx]


def get_dim_and_dir(contexts, n_dims, context_size, one_hot=False):
    """Gets the dimension and whether it's max/min on that dimension for the
    first object in a batch of contexts. """
    batch_size = np.shape(contexts)[0]
    dims, mins = np.zeros(batch_size), np.zeros(batch_size)
    for dim in range(n_dims):
        the_dim = contexts[np.arange(batch_size)[:, None],
                           np.repeat(np.arange(dim, context_size*n_dims + dim,
                                               n_dims),
                                     batch_size, axis=0)]
        min_dim = np.argmin(the_dim, axis=1)
        min_dim = (min_dim == 0).astype(int)  # where non-zero
        max_dim = np.argmax(the_dim, axis=1)
        max_dim = (max_dim == 0).astype(int)
        mins += max_dim
        dims += dim*(min_dim + max_dim)

    if one_hot:
        dims = np.eye(n_dims)[dims.astype(int)]
        mins = np.eye(2)[mins.astype(int)]

    return dims, mins


if __name__ == '__main__':

    # TODO: argparse stuff!
    n_dims = 4
    objs = np.arange(0, 1, 1/20)
    # TODO: vary context size, not just 2*NDIMS...
    context_size = 2 * n_dims  # number of objects
    fixed_sender = True

    batch_size = 16
    num_batches = 50000

    if not fixed_sender:
        sender = Sender(context_size, n_dims)
        sender_opt = torch.optim.Adam(sender.parameters())

    receiver = Receiver(context_size, n_dims)
    receiver_opt = torch.optim.Adam(receiver.parameters())

    writer = SummaryWriter()

    for batch in range(num_batches):

        # 1. get contexts from Nature
        contexts = np.stack([get_context(n_dims, objs)
                            for _ in range(batch_size)])
        # batch normalize?
        # TODO: batch normalize each _dimension_ before combining into
        # context instead of whole context??
        # contexts = (contexts - np.mean(contexts)) / (np.std(contexts) + 1e-12)

        # 1a. permute context for receiver
        # NOTE: sender always sends 'first' object in context; receiver sees
        # permuted context
        rec_perms = get_permutations(batch_size, context_size)
        rec_contexts = apply_perms(contexts, rec_perms, n_dims, batch_size)
        # 1b. get correct target based on perms
        target = np.zeros(shape=(batch_size, 1), dtype=np.int64)
        rec_target = rec_perms[np.arange(batch_size)[:, None], target]

        # 2. get signals form sender
        if fixed_sender:
            dim_msg, min_msg = get_dim_and_dir(contexts, n_dims, context_size,
                                               one_hot=True)
            dim_msg = torch.Tensor(dim_msg)
            min_msg = torch.Tensor(min_msg)
        else:
            dim_probs, min_probs = sender(torch.Tensor(contexts))
            dim_dist = torch.distributions.OneHotCategorical(dim_probs)
            dim_msg = dim_dist.sample()
            min_dist = torch.distributions.OneHotCategorical(min_probs)
            min_msg = min_dist.sample()

        # 3. get choice from receiver
        choice_probs = receiver(torch.Tensor(rec_contexts), dim_msg, min_msg)
        """
        choice_dist = torch.distributions.Categorical(choice_probs)
        choice = choice_dist.sample()
        print(choice)

        # 4. get reward
        reward = torch.unsqueeze(
            torch.eq(
                torch.from_numpy(rec_target.flatten()),
                choice).float(),
            dim=0).detach()
        # reward 1/0 goes to -1/1
        # advantages = reward
        advantages = 2*reward - 1
        # advantages = (reward - reward.mean()) / (reward.std() + 1e-12)
        """

        # 5. compute losses and reinforce

        # 5a. sender
        if not fixed_sender:
            dim_log_prob = dim_dist.log_prob(dim_msg)
            min_log_prob = min_dist.log_prob(min_msg)
            sender_opt.zero_grad()
            sender_loss = -torch.sum(advantages * (dim_log_prob + min_log_prob))
            sender_loss.backward()
            sender_opt.step()

        # 5b. receiver
        # choice_log_prob = choice_dist.log_prob(choice)
        receiver_opt.zero_grad()
        # receiver_loss = -torch.sum(advantages * choice_log_prob)
        # receiver_loss = F.cross_entropy(choice_probs, torch.Tensor(rec_target.flatten()).long())
        receiver_loss = F.mse_loss(choice_probs, torch.Tensor(
            contexts[np.arange(batch_size)[:, None],
                     np.repeat(np.arange(n_dims)[None, :],
                               batch_size, axis=0)]
        ))
        receiver_loss.backward()
        receiver_opt.step()

        print('\nIteration: {}'.format(batch))
        print(contexts)
        print(torch.cat([dim_msg, min_msg], dim=1))
        print(choice_probs)
        print(receiver_loss)
        """
        print(reward)
        print('% correct: {}'.format(torch.mean(reward)))
        """
