import tensornetwork as tn
import pytest
import numpy as np

def test_contextmanager_simple():

    tn.set_default_backend("pytorch")

    with tn.DefaultBackend("numpy"):
        a = tn.Node(np.ones((10,)))
        b = tn.Node(np.ones((10,)))

    assert (type(a.backend) == type(b.backend)) == True
    edge = a[0] ^ b[0] # Equal to tn.connect(a[0], b[0])
    final_node = tn.contract(edge)
    assert final_node.tensor == 10.0


def test_contextmanager_interaption():
    tn.set_default_backend("pytorch")
    assert tn.config.default_backend == "pytorch"

    with tn.DefaultBackend("numpy"):

        assert tn.config.default_backend == "pytorch"

        a = tn.Node(np.ones((10,)))
        tn.set_default_backend("tensorflow")
        b = tn.Node(np.ones((10,)))

    assert (type(a.backend) == type(b.backend)) == True
    assert tn.config.default_backend == "tensorflow"
