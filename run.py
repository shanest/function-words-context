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
import tensorflow as tf

tf.enable_eager_execution()

# TODO: major refactor, make everything modular!!

NDIMS = 1
OBJS = np.arange(0, 12)
CONTEXT_SIZE = 2*NDIMS
DIM_MESSAGE = int(NDIMS > 1)

# TODO: vary context size, not just 2*NDIMS...
# TODO: parameterize hidden layers
sender = tf.keras.Sequential([
    tf.keras.layers.Dense(32, input_shape=(NDIMS*CONTEXT_SIZE + CONTEXT_SIZE,),
                          activation=tf.nn.elu),
    tf.keras.layers.Dense(16, activation=tf.nn.elu),
    tf.keras.layers.Dense(2 + NDIMS*DIM_MESSAGE)
])

receiver = tf.keras.Sequential([
    tf.keras.layers.Dense(32,
                          input_shape=(CONTEXT_SIZE*NDIMS + NDIMS*DIM_MESSAGE + 2,),
                          activation=tf.nn.elu),
    tf.keras.layers.Dense(16, activation=tf.nn.elu),
    tf.keras.layers.Dense(CONTEXT_SIZE)
])

BATCH_SIZE = 16
NUM_BATCHES = 100000

optimizer = tf.train.AdamOptimizer(1e-5)


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
    """
    objs = np.array(objs)
    objs = np.transpose(objs)
    np.random.shuffle(objs)
    """
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

    for batch in range(NUM_BATCHES):

        # TODO: sender and receiver see SAME context, but in different _order_!
        # sender perm, receiver perm; get them until they are diff
        # apply to context, which is still ordered...

        # 1. get contexts from Nature
        context = np.stack([get_context(NDIMS, OBJS)
                            for _ in range(BATCH_SIZE)])

        sender_perms = get_permutations(BATCH_SIZE, CONTEXT_SIZE)
        sender_contexts = apply_perms(context, sender_perms, NDIMS, BATCH_SIZE)

        rec_perms = get_permutations(BATCH_SIZE, CONTEXT_SIZE)
        rec_contexts = apply_perms(context, rec_perms, NDIMS, BATCH_SIZE)

        target = np.random.randint(CONTEXT_SIZE, size=(BATCH_SIZE, 1))
        sender_target = sender_perms[np.arange(BATCH_SIZE)[:, None], target]
        rec_target = rec_perms[np.arange(BATCH_SIZE)[:, None], target]

        target_one_hot = tf.squeeze(tf.one_hot(sender_target, depth=CONTEXT_SIZE))

        rows = np.reshape(np.arange(BATCH_SIZE), (BATCH_SIZE, 1))
        obj_lens = np.stack([np.arange(NDIMS) for _ in range(BATCH_SIZE)])
        target_values = tf.to_float(context[rows, target+obj_lens])

        with tf.GradientTape() as tape:
            # 2. get signal(s) from sender
            all_message_logits = sender(
                tf.concat([context, target_one_hot], axis=1))
            min_max_message_logits = all_message_logits[:, -2:]
            min_max_message = tf.stop_gradient(tf.squeeze(tf.one_hot(
                tf.multinomial(min_max_message_logits, num_samples=1),
                depth=2)))
            if NDIMS > 1:
                dim_message_logits = all_message_logits[:, :-2]
                dim_message = tf.stop_gradient(tf.squeeze(tf.one_hot(
                    tf.multinomial(dim_message_logits, num_samples=1),
                    depth=NDIMS)))
                message = tf.concat([dim_message, min_max_message], axis=1)
            else:
                message = min_max_message

            # 3. get choice from receiver
            choice_logits = receiver(tf.concat([context, message], axis=1))
            choice = tf.stop_gradient(tf.squeeze(tf.one_hot(
                tf.multinomial(choice_logits, num_samples=1),
                depth=CONTEXT_SIZE)))

            # 4. get reward
            # TODO: record awards over time, or running mean, or....
            reward = tf.stop_gradient(tf.to_float(
                tf.equal(tf.squeeze(rec_target), tf.argmax(choice, axis=1))))
            # reward 1/0 goes to -1/1; minimize loss, not maximize prob
            # advantages = 2*reward - 1
            # TODO: why does this work better than advantages?
            advantages = reward

            # 5. compute losses
            sender_loss = tf.reduce_mean(
                advantages * tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.squeeze(tf.argmax(min_max_message, axis=1)),
                    logits=min_max_message_logits))
            if NDIMS > 1:
                sender_loss = sender_loss + tf.reduce_mean(
                    advantages * tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=tf.squeeze(tf.argmax(dim_message, axis=1)),
                        logits=dim_message_logits))

            receiver_loss = tf.reduce_mean(
                advantages * tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.squeeze(tf.argmax(choice, axis=1)),
                    logits=choice_logits))

            grads = tape.gradient(sender_loss + receiver_loss,
                                  sender.variables + receiver.variables)

        print('\nIteration: {}'.format(batch))
        print(context)
        print(target)
        print(message)
        print(reward)
        print('Mean reward: {}'.format(tf.reduce_mean(reward)))
        optimizer.apply_gradients(zip(grads, sender.variables + receiver.variables),
                           global_step=tf.train.get_or_create_global_step())
