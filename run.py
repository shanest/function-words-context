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
OBJS = list(range(12))
CONTEXT_SIZE = 2
DIM_MESSAGE = int(NDIMS > 1)

# TODO: vary context size, not just 2*NDIMS...
# TODO: parameterize hidden layers
sender = tf.keras.Sequential([
    tf.keras.layers.Dense(8, input_shape=(NDIMS*(CONTEXT_SIZE + 1),),
                          activation=tf.nn.elu),
    tf.keras.layers.Dense(8, activation=tf.nn.elu),
    tf.keras.layers.Dense(2 + NDIMS*DIM_MESSAGE)
])

receiver = tf.keras.Sequential([
    tf.keras.layers.Dense(8, input_shape=(CONTEXT_SIZE*(NDIMS + 1),),
                          activation=tf.nn.elu),
    tf.keras.layers.Dense(8, activation=tf.nn.elu),
    tf.keras.layers.Dense(CONTEXT_SIZE)
])

BATCH_SIZE = 16
NUM_BATCHES = 100000

optimizer = tf.train.AdamOptimizer(1e-4)


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
        min_idx = idx * n_dims
        dimension[[where_min, min_idx]] = dimension[[min_idx, where_min]]
        where_max = np.argmax(dimension)
        max_idx = min_idx + 1
        dimension[[where_max, max_idx]] = dimension[[max_idx, where_max]]
        objs.append(dimension)
    objs = np.array(objs)
    objs = np.transpose(objs)
    np.random.shuffle(objs)
    return objs.flatten()


if __name__ == '__main__':

    for _ in range(NUM_BATCHES):

        # 1. get contexts from Nature
        # TODO: get_context method, for >1 dim
        context = np.stack([np.random.choice(OBJS, size=CONTEXT_SIZE, replace=False)
                            for _ in range(BATCH_SIZE)])
        target = np.random.randint(CONTEXT_SIZE, size=(BATCH_SIZE))
        target_one_hot = tf.one_hot(target, depth=CONTEXT_SIZE)
        # TODO: get entire obj_slice, not just single value, when NDIMS > 1
        rows = tf.range(BATCH_SIZE)
        indices = tf.stack([rows, target], axis=1)
        target_values = tf.to_float(tf.reshape(
            tf.gather_nd(context, indices),
            (BATCH_SIZE, 1)))

        with tf.GradientTape() as tape:
            # 2. get signal(s) from sender
            # TODO: make this work with length-2 signals
            message_logits = sender(
                tf.concat([context, target_values], axis=1))
            message = tf.stop_gradient(tf.squeeze(tf.one_hot(
                tf.multinomial(message_logits, num_samples=1),
                depth=CONTEXT_SIZE)))

            # 3. get choice from receiver
            choice_logits = receiver(tf.concat([context, message], axis=1))
            choice = tf.stop_gradient(tf.squeeze(tf.one_hot(
                tf.multinomial(choice_logits, num_samples=1),
                depth=CONTEXT_SIZE)))

            # 4. get reward
            # TODO: record awards over time, or running mean, or....
            reward = tf.stop_gradient(tf.to_float(
                tf.equal(target, tf.argmax(choice, axis=1))))
            # reward 1/0 goes to 1/-1
            advantages = 2*reward - 1

            # 5. compute losses
            sender_loss = tf.reduce_mean(
                advantages * tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=message,
                    logits=message_logits))

            receiver_loss = tf.reduce_mean(
                advantages * tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=choice,
                    logits=choice_logits))

            grads = tape.gradient(sender_loss + receiver_loss,
                                  sender.variables + receiver.variables)

        print('')
        print(context)
        print(target)
        print(message)
        print(reward)
        print('Mean reward: {}'.format(tf.reduce_mean(reward)))
        optimizer.apply_gradients(zip(grads, sender.variables + receiver.variables),
                           global_step=tf.train.get_or_create_global_step())
