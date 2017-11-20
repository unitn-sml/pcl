
import pickle
import numpy as np

from textwrap import dedent


def dump(path, what):
    with open(path, 'wb') as fp:
        pickle.dump(what, fp, protocol=pickle.HIGHEST_PROTOCOL)


def load(path):
    try:
        with open(path, 'rb') as fp:
            return pickle.load(fp)
    except FileNotFoundError:
        return None


def subdict(d, keys=None, nokeys=None):
    """Returns a subdictionary.

    Parameters
    ----------
    d : dict
        A dictionary.
    keys : list or set
        The set of keys to include in the subdictionary. If None use all keys.
    nokeys : list or set
        The set of keys to not include in the subdictionary. If None use no keys.
    """
    keys = set(keys if keys else d.keys())
    nokeys = set(nokeys if nokeys else [])
    return {k: v for k, v in d.items() if k in (keys - nokeys)}


def freeze(x):
    """Freezes a dictionary, i.e. makes it immutable and thus hashable."""
    if x is None:
        return None
    if isinstance(x, (list, np.ndarray)):
        return tuple(x)
    elif isinstance(x, dict):
        frozen = {}
        for k, v in sorted(x.items()):
            if isinstance(v, (list, np.ndarray)):
                frozen[k] = tuple(v)
            else:
                frozen[k] = v
        return frozenset(frozen.items())
    raise ValueError('Cannot freeze objects of type {}'.format(type(x)))


def _phi(template):
    return template + '\nsolve satisfy;'


def _infer(template, feat_var='phi', weights_var='w'):
    infer = dedent('''
        array[FEATURES] of int: {weights_var};
        var int: utility = sum(i in index_set({feat_var}))({weights_var}[i] * {feat_var}[i]);
        solve maximize utility;
    ''').format(**locals())
    return template + infer


def _improve(template, improve_margin, feat_var='phi', weights_var='w'):
    improve = dedent('''
        int: improve_margin = {improve_margin};
        array[FEATURES] of int: {weights_var};
        var int: utility = sum(i in index_set({feat_var}))({weights_var}[i] * {feat_var}[i]);
        constraint utility > improve_margin;
        solve minimize utility;
    ''').format(**locals())
    return template + improve

