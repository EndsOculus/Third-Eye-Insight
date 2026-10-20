"""Exercise the native empty-PackedSequence fix in an isolated CPU process."""
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize('operation', ['unpack_sequence', 'pad_packed_sequence'])
def test_empty_packed_sequence_raises_instead_of_crashing(operation):
    # Keep a vulnerable native build from crashing the pytest host process.
    code = '''
from importlib.metadata import version
from packaging.version import Version
import sys

assert Version(version('torch')) >= Version('2.10.0'), 'The tested Windows fix requires torch >=2.10.0'
import torch
from torch.nn.utils import rnn

operation = getattr(rnn, sys.argv[1])
packed = rnn.PackedSequence(torch.empty(0, 5), torch.empty(0, dtype=torch.int64))
try:
    operation(packed)
except RuntimeError as error:
    assert 'batch_sizes' in str(error) and 'empty' in str(error)
else:
    raise AssertionError('An empty PackedSequence must be rejected')

valid = rnn.pack_sequence([torch.ones(2, 5), torch.ones(1, 5)], enforce_sorted=False)
if sys.argv[1] == 'unpack_sequence':
    result = operation(valid)
    assert [tuple(item.shape) for item in result] == [(2, 5), (1, 5)]
else:
    padded, lengths = operation(valid, batch_first=True)
    assert tuple(padded.shape) == (2, 2, 5) and lengths.tolist() == [2, 1]
print(torch.__version__)
'''
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    result = subprocess.run([sys.executable, '-B', '-c', code, operation],
                            capture_output=True, text=True, timeout=60, env=environment)
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
