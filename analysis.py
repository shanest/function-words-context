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
import pandas as pd
from plotnine import *


def visualize_trial(data):

    data = data.astype('category')  # just in case
    msg_vars = [col for col in data if col.startswith('msg')]
    long_data = pd.melt(
        data,
        id_vars=['true_dim', 'true_mins', 'true_total', 'total_msg', 'correct'],
        value_vars=msg_vars,
        var_name='msg',
        value_name='sent')
    long_data = pd.melt(
        long_data,
        id_vars=['true_total', 'total_msg', 'correct', 'msg', 'sent'],
        value_vars=['true_dim', 'true_mins'],
        var_name='feature',
        value_name='value')

    print(ggplot(data=long_data) +
          geom_bar(aes('value', fill='sent'), position='dodge') +
          facet_wrap(['feature', 'msg']))


def visualize_training(base_dir='../data/exp1/',
                       dims=range(1, 4), trials=range(10)):

    tall_data = pd.DataFrame()
    for dim in dims:
        for trial in trials:
            cur = pd.read_csv('{}/n{}/trial_{}/train.csv'.format(base_dir,
                                                                 dim, trial))
            cur['trial'] = (dim-1)*len(trials) + trial
            cur['dims'] = dim
            cur['accuracy'] = cur.rolling(
                10, min_periods=1).mean()['percent_correct']
            tall_data = tall_data.append(cur)
    tall_data['trial'] = tall_data['trial'].astype('category')
    tall_data['dims'] = tall_data['dims'].astype('category')
    print(ggplot(data=tall_data) +
          geom_line(aes(x='batch_num', y='accuracy',
                        group='trial',
                        colour='dims')))
