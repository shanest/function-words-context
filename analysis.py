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
        id_vars=['true_dim', 'true_mins', 'true_total', 'total_msg'],
        value_vars=msg_vars,
        var_name='msg',
        value_name='sent')

    print(ggplot(data=long_data) +
     geom_bar(aes('true_dim', fill='sent'), position='dodge') +
     facet_wrap('msg'))

    print(ggplot(data=long_data) +
     geom_bar(aes('true_mins', fill='sent'), position='dodge') +
     facet_wrap('msg'))
