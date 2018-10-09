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

import random
import tensorflow as tf

tf.enable_eager_execution()

NDIMS = 1
OBJS = list(range(100))
CONTEXT_SIZE = 2
DIM_MESSAGE = int(NDIMS > 1)

# TODO: vary context size, not just 2*NDIMS...
# TODO: parameterize hidden layers
sender = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(CONTEXT_SIZE*(NDIMS + 1),),
                          activation=tf.nn.elu),
    tf.keras.layers.Dense(2 + NDIMS*DIM_MESSAGE)
])

receiver = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(CONTEXT_SIZE*(NDIMS + 1),),
                          activation=tf.nn.elu),
    tf.keras.layers.Dense(CONTEXT_SIZE)
])

BATCH_SIZE = 8
NUM_BATCHES = 10

for _ in range(NUM_BATCHES):

    context = [random.sample(OBJS, CONTEXT_SIZE) for _ in range(BATCH_SIZE)]
    target = [random.randrange(CONTEXT_SIZE) for _ in range(BATCH_SIZE)]
    print(target)
    # TODO: make this work with length-2 signals
    message_probs = sender(
        tf.concat([context, tf.one_hot(target, depth=CONTEXT_SIZE)], axis=1))
    message = tf.one_hot(tf.argmax(message_probs, axis=1), depth=CONTEXT_SIZE)
    print(message_probs)
    print(tf.one_hot(tf.argmax(message, axis=1), depth=CONTEXT_SIZE))
    choice_logits = receiver(tf.concat([context, message], axis=1))
    print(choice_logits)
    print(tf.to_int64(tf.equal(target, tf.argmax(choice_logits, axis=1))))
