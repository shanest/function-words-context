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
import context


def unzip(ls):
    """Takes a list of pairs and returns a pair of lists. """
    return list(zip(*ls))


def dirs_and_dims(contexts):
    return unzip([context.dir_and_dim() for context in contexts])


def get_context_size(n_dims, scale, n_objs, at_dim_idx):
    dummy = context.Context(n_dims, scale, n_objs)
    return len(dummy.view())
