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
import models


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


def get_communicative_success(contexts, objs, n_dims):
    """Checks whether a predicted object (as values in each dimension) is
    closer to the first object in context or not. """
    n_objs = int(np.shape(contexts)[1] / np.shape(objs)[1])
    context_objs = np.reshape(contexts, (-1, n_dims))
    repeat_objs = np.repeat(objs, n_objs, axis=0)
    mse_per_obj = np.reshape(
        np.mean((context_objs - repeat_objs)**2, axis=1),
        (-1, n_objs))
    # dot_per_obj = np.sum(context_objs * repeat_objs, axis=1)[:, None]
    predicted_obj = np.argmin(mse_per_obj, axis=1)[:, None]
    return (predicted_obj == 0).astype(int)


if __name__ == '__main__':

    # TODO: argparse stuff!
    n_dims = 2
    objs = np.arange(-1, 1, 1/5)
    # TODO: vary context size, not just 2*NDIMS...
    context_size = 2 * n_dims  # number of objects
    fixed_sender = False

    batch_size = 32
    num_batches = 50000

    if not fixed_sender:
        sender = models.Sender(context_size, n_dims)
        sender_opt = torch.optim.Adam(sender.parameters())

    receiver = models.Receiver(context_size, n_dims)
    receiver_opt = torch.optim.Adam(receiver.parameters())

    for batch in range(num_batches):

        # 1. get contexts and target object from Nature
        contexts = np.stack([get_context(n_dims, objs)
                            for _ in range(batch_size)])
        target_obj = torch.Tensor(
            contexts[np.arange(batch_size)[:, None],
                     np.repeat(np.arange(n_dims)[None, :],
                               batch_size, axis=0)]
        )

        # 1a. permute context for receiver
        # NOTE: sender always sends 'first' object in context; receiver sees
        # permuted context
        rec_perms = get_permutations(batch_size, context_size)
        rec_contexts = apply_perms(contexts, rec_perms, n_dims, batch_size)
        # 1b. get correct target index based on perms
        target = np.zeros(shape=(batch_size, 1), dtype=np.int64)
        rec_target = np.where(rec_perms == 0)[1][:, None]

        # 2. get signals form sender
        if fixed_sender:
            msgs = get_dim_and_dir(contexts, n_dims, context_size, one_hot=True)
            msgs_in = torch.cat([torch.Tensor(msg) for msg in msgs], dim=1)
        else:
            msg_probs = sender(torch.Tensor(contexts))
            msg_dists = [torch.distributions.OneHotCategorical(probs)
                         for probs in msg_probs]
            msgs = [dist.sample() for dist in msg_dists]
            msgs_in = torch.cat(msgs, dim=1)

        # 3. get choice from receiver
        choice_objs, choice_probs = receiver(torch.Tensor(rec_contexts),
                                             msgs_in)
        choice_dist = torch.distributions.Categorical(choice_probs)
        choice = choice_dist.sample()

        # 4. get reward
        reward = torch.eq(
                torch.from_numpy(rec_target.flatten()),
                choice).float().detach()
        advantages = reward
        # reward 1/0 goes to -1/1
        # advantages = 2*reward - 1
        # advantages = reward - reward.mean() / (reward.std() + 1e-8)

        # 5. compute losses and reinforce

        # 5a. sender
        if not fixed_sender:
            sender_opt.zero_grad()
            msg_log_probs = [msg_dists[idx].log_prob(msgs[idx])
                             for idx in range(len(msgs))]
            sender_loss = -torch.sum(
                advantages *
                torch.sum(torch.stack(msg_log_probs, dim=1), dim=1))
            sender_loss.backward()
            sender_opt.step()

        # 5b. receiver
        receiver_opt.zero_grad()
        choice_log_prob = choice_dist.log_prob(choice)
        receiver_reinforce = -torch.sum(advantages * choice_log_prob)
        receiver_mse = F.mse_loss(choice_objs, target_obj)
        receiver_loss = receiver_reinforce
        receiver_loss.backward()
        receiver_opt.step()

        if batch % 50 == 0:
            print('\nIteration: {}'.format(batch))
            print(contexts)
            # print(choice_objs)
            print(msgs_in)
            print(receiver_mse)
            print(reward)
            print('% correct: {}'.format(torch.mean(reward)))
