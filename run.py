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
import pandas as pd
import models
import context
import util


def run_trial(num, out_dir, sender_fn=None, receiver_fn=None,
              n_dims=2, scale=np.arange(-1, 1, 1/10),
              dim_first=False, at_dim_idx=False,
              batch_size=32, num_batches=15000, record_every=50,
              save_models=True, num_test=5000, **kwargs):

    n_objs = 2*n_dims  # TODO: modify get_context, allow to vary
    context_size = util.get_context_size(n_dims, scale, n_objs, at_dim_idx)
    data = pd.DataFrame(columns=['batch_num', 'percent_correct'])

    if sender_fn is not None:
        sender = sender_fn(context_size, n_dims)
        sender_opt = torch.optim.Adam(sender.parameters(), lr=5e-4)

    # TODO: generalize max_msg argument to receivers
    receiver = receiver_fn(context_size, n_dims, n_objs)
    receiver_opt = torch.optim.Adam(receiver.parameters(), lr=5e-4)

    def one_batch(batch_size):
        # 1. get contexts and target object from Nature
        contexts = [context.Context(n_dims, scale) for _ in range(batch_size)]
        # contexts = contexts - 0.5

        # 1a. permute context for receiver
        # NOTE: sender always sends 'first' object in context; receiver sees
        # permuted context
        rec_perms = [np.random.permutation(n_objs) for _ in range(batch_size)]
        rec_dims = [contexts[idx].permuted_dims(rec_perms[idx])
                    for idx in range(len(rec_perms))]
        # 1b. get correct target index based on perms
        rec_target = np.where(np.array(rec_perms) == 0)[1][:, None]

        # 2. get signals form sender
        if sender_fn is None:
            # TODO: implement FixedSender as a nn.Module in models, so that the
            # code can be maximally modular?  Would require returning
            # ``probabilities'' and wasting compute time ``training'' it
            # TODO: fix this with new dir_and_dim; get one hots
            """
            msgs = [torch.Tensor(val) for val in
                    get_dim_and_dir(contexts, n_dims, context_size, one_hot=True)]
            """
            msg_dists = None
        else:
            sender_contexts = torch.Tensor([con.view(dim_first=dim_first,
                                                     at_dim_idx=at_dim_idx)
                                            for con in contexts])
            msg_dists, msgs = sender(sender_contexts)

        # 3. get choice from receiver
        rec_contexts = [contexts[idx].view(dims=rec_dims[idx],
                                           dim_first=dim_first,
                                           at_dim_idx=at_dim_idx)
                        for idx in range(len(rec_dims))]
        choice_dists, choices = receiver(torch.Tensor(rec_contexts), msgs)
        """
        choice_probs = receiver(torch.Tensor(rec_contexts), msgs)
        choice_dist = torch.distributions.Categorical(choice_probs)
        choice = choice_dist.sample()
        """
        # true_dims, _ = get_dim_and_dir(contexts, n_dims, context_size)

        # 4. get reward
        reward = torch.eq(torch.from_numpy(rec_target.flatten()),
                          choices[-1]).float().detach()

        return contexts, msg_dists, msgs, choice_dists, choices, reward

    for batch in range(num_batches):
        contexts, msg_dists, msgs, choice_dists, choices, reward = \
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
        """
        choice_log_prob = choice_dist.log_prob(choice)
        receiver_reinforce = -torch.sum(advantages * choice_log_prob)
        """
        choice_log_probs = [choice_dists[idx].log_prob(choices[idx])
                            for idx in range(len(choices))]
        receiver_loss = -torch.sum(
            advantages *
            torch.sum(torch.stack(choice_log_probs, dim=1), dim=1))
        # receiver_loss = receiver_reinforce
        receiver_loss.backward()
        receiver_opt.step()

        if batch % record_every == 0:
            print('\nIteration: {}'.format(batch))
            print(np.array([con.view(at_dim_idx=at_dim_idx, dim_first=dim_first)
                   for con in contexts]))
            print(torch.cat(msgs, dim=1))
            print(choices[0])
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
        true_mins, true_dims = util.dirs_and_dims(contexts)
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
    parser.add_argument('--at_dim_idx', action='store_true')
    parser.add_argument('--dim_first', action='store_true')
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
