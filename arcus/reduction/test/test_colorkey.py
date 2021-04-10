import numpy as np
from ..ogip import ColOrKeyTable

import pytest

@pytest.mark.parametrize('input', [3, [3, 3], 3 * np.ones(2, dtype=int)])
def test_insert_allsame(input):
    t = ColOrKeyTable([['text1', 'text2'], [1, 2]], names=['a', 'b'])
    t['c'] = input
    assert t.meta['c'] == 3
    assert t['c'] == 3
    assert t.colnames == ['a', 'b']


def test_access_meta_as_column():
    t = ColOrKeyTable([['text1', 'text2'], [1, 2]], names=['a', 'b'])
    t.meta['test'] = 'mytext'
    assert t['test'] == 'mytext'


def test_existing_col_allsame(input):
    t = ColOrKeyTable([['text1', 'text2'], [1, 2]], names=['a', 'b'])
    t['a'] = 'text3'
    assert t.meta['a'] == 'text3'
    assert t['a'] == 'text3'
    assert t.colnames == ['b']


def test_existing_col_allsame_slice(input):
    t = ColOrKeyTable([['text1', 'text2'], [1, 2]], names=['a', 'b'])
    t['a'][1] = 'text1'
    assert t.meta['a'] == 'text3'
    assert t['a'] == 'text3'
    assert t.colnames == ['b']
