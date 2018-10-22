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
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import models


def get_context(n_dims, scale, with_dim_label=False):
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

    if with_dim_label:
        dims = np.concatenate([np.eye(n_dims)]*2*n_dims)
        objs = np.reshape(objs, (-1, 1))
        objs = np.concatenate([dims, objs], axis=1)
        objs = objs.reshape((2*n_dims, -1))

    np.random.shuffle(objs)
    return objs.flatten()


def get_permutations(batch_size, context_size):
    return np.stack([np.random.permutation(context_size)
                     for _ in range(batch_size)])


def apply_perms(contexts, perms, n_dims, batch_size, with_dim_labels):
    # TODO: DOCUMENT
    obj_indices = np.tile(np.stack(
        [np.arange(n_dims + int(with_dim_labels)*n_dims**2)
         for _ in range(batch_size)]), 2*n_dims)
    perm_idx = (np.repeat(perms, n_dims + int(with_dim_labels)*n_dims**2, axis=1)
                * (n_dims + int(with_dim_labels)*n_dims**2) + obj_indices)
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

    return dims.astype(int), mins.astype(int)


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


def run_trial(num, out_dir, sender_fn=None, receiver_fn=None,
              n_dims=2, objs=np.arange(-1, 1, 1/100), with_dim_labels=False,
              batch_size=32, num_batches=15000, record_every=50,
              save_models=True, num_test=5000, **kwargs):

    context_size = 2*n_dims  # TODO: modify get_context, allow to vary
    data = pd.DataFrame(columns=['batch_num', 'percent_correct'])

    if sender_fn is not None:
        sender = sender_fn(context_size, n_dims, with_dim_labels)
        sender_opt = torch.optim.Adam(sender.parameters())

    # TODO: generalize max_msg argument to receivers
    receiver = receiver_fn(context_size, n_dims, n_dims, with_dim_labels)
    receiver_opt = torch.optim.Adam(receiver.parameters())

    def one_batch(batch_size):
        # 1. get contexts and target object from Nature
        contexts = np.stack([get_context(n_dims, objs, with_dim_labels)
                            for _ in range(batch_size)])

        # 1a. permute context for receiver
        # NOTE: sender always sends 'first' object in context; receiver sees
        # permuted context
        rec_perms = get_permutations(batch_size, context_size)
        rec_contexts = apply_perms(contexts, rec_perms, n_dims, batch_size,
                                   with_dim_labels)
        # 1b. get correct target index based on perms
        rec_target = np.where(rec_perms == 0)[1][:, None]

        # 2. get signals form sender
        if sender_fn is None:
            # TODO: implement FixedSender as a nn.Module in models, so that the
            # code can be maximally modular?  Would require returning
            # ``probabilities'' and wasting compute time ``training'' it
            msgs = [torch.Tensor(val) for val in
                    get_dim_and_dir(contexts, n_dims, context_size, one_hot=True)]
            msg_dists = None
        else:
            msg_dists, msgs = sender(torch.Tensor(contexts))

        # 3. get choice from receiver
        choice_probs = receiver(torch.Tensor(rec_contexts), msgs)
        choice_dist = torch.distributions.Categorical(choice_probs)
        choice = choice_dist.sample()
        true_dims, _ = get_dim_and_dir(contexts, n_dims, context_size)

        # 4. get reward
        reward = torch.eq(torch.from_numpy(rec_target.flatten()),
                          choice).float().detach()

        return contexts, msg_dists, msgs, choice_dist, choice, reward

    for batch in range(num_batches):
        contexts, msg_dists, msgs, choice_dist, choice, reward = \
                one_batch(batch_size)
        advantages = reward
        # reward 1/0 goes to -1/1
        # advantages = 2*reward - 1
        # advantages = reward - reward.mean() / (reward.std() + 1e-8)

        # 5. compute losses and reinforce
        # 5a. sender
        if sender_fn is not None:
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
        receiver_loss = receiver_reinforce
        receiver_loss.backward()
        receiver_opt.step()

        if batch % record_every == 0:
            print('\nIteration: {}'.format(batch))
            print(contexts)
            print(torch.cat(msgs, dim=1))
            print(reward)
            percent = torch.mean(reward).data.item()
            print('% correct: {}'.format(percent))
            data = data.append(
                {'batch_num': batch, 'percent_correct': percent},
                ignore_index=True)

    out_root = '{}/trial_{}/'.format(out_dir, trial)
    if not os.path.exists(out_root):
        os.makedirs(out_root)

    data.to_csv(out_root + 'train.csv')

    if save_models:
        torch.save(receiver.state_dict(), out_root + 'receiver.pt')
        if sender_fn is not None:
            torch.save(sender.state_dict(), out_root + 'sender.pt')

    if num_test:
        contexts, _, msgs, _, choice, reward = one_batch(num_test)
        true_dims, true_mins = get_dim_and_dir(contexts, n_dims, context_size)
        # TODO: record more? whole context, other features of it?
        test_data = pd.DataFrame({'true_dim': true_dims,
                                  'true_mins': true_mins,
                                  'correct': reward.numpy().astype(int)})
        test_data['true_total'] = (test_data['true_dim'] +
                                   n_dims*test_data['true_mins'])
        print('Test reward: {}'.format(test_data['correct'].mean()))
        for idx in range(len(msgs)):
            test_data['msg_' + str(idx)] = np.argmax(msgs[idx].numpy(), axis=1)
        # TODO: document!
        msg_col = test_data['msg_0']
        for idx in range(1, len(msgs)):
            msg_col = (msg_col +
                       (test_data['msg_'+str(idx-1)].max() + 1) *
                       test_data['msg_'+str(idx)])
        test_data['total_msg'] = msg_col
        test_data = test_data.astype('category')
        test_data.to_csv(out_root + 'test.csv')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_trials', type=int, default=1)
    parser.add_argument('--out_path', type=str, default='../data/')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_batches', type=int, default=50000)
    parser.add_argument('--record_every', type=int, default=50)
    parser.add_argument('--num_test', type=int, default=5000)
    parser.add_argument('--n_dims', type=int, default=2)
    parser.add_argument('--sender_type', type=str, default=None)
    parser.add_argument('--receiver_type', type=str, default='base')
    parser.add_argument('--with_dim_labels', action='store_true')
    args = parser.parse_args()

    args.sender_fn = {
        'base': models.Sender,
        'split': models.SplitSender,
        'rnn': models.RNNSender
    }.get(args.sender_type)  # .get allows default None for fixed_sender

    args.receiver_fn = {
        'base': models.BaseReceiver,
        'dim': models.DimReceiver,
        'mse': models.MSEReceiver,
        'rnn': models.RNNReceiver
    }[args.receiver_type]

    for trial in range(args.num_trials):
        run_trial(trial, args.out_path, **vars(args))
