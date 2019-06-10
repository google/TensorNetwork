import collections
import tensorflow as tf
""" 
@author: Markus Hauru
A module for the function ncon, which does contractions of several tensors.
"""

def ncon(AA, v, order=None, forder=None, check_indices=True):

    """ AA = [A1, A2, ..., Ap] list of tensors.

    v = (v1, v2, ..., vp) tuple of lists of indices e.g. v1 = [3 4 -1] labels
    the three indices of tensor A1, with -1 indicating an uncontracted index
    (open leg) and 3 and 4 being the contracted indices.

    order, if present, contains a list of all positive indices - if not
    [1 2 3 4 ...] by default. This is the order in which they are contracted.

    forder, if present, contains the final ordering of the uncontracted indices
    - if not, [-1 -2 ..] by default.

    There is some leeway in the way the inputs are given. For example,
    instead of giving a list of tensors as the first argument one can
    give some different iterable of tensors, such as a tuple, or a
    single tensor by itself (anything that has the attribute "shape"
    will be considered a tensor).
    """

    # We want to handle the tensors as a list, regardless of what kind
    # of iterable we are given. In addition, if only a single element is
    # given, we make list out of it. Inputs are assumed to be non-empty.
    if hasattr(AA, "shape"):
        AA = [AA]
    else:
        AA = list(AA)
    v = list(v)
    if not isinstance(v[0], collections.Iterable):
        # v is not a list of lists, so make it such.
        v = [v]
    else:
        v = list(map(list, v))

    if order == None:
        order = create_order(v)
    if forder == None:
        forder = create_forder(v)

    if check_indices:
        # Raise a RuntimeError if the indices are wrong.
        do_check_indices(AA, v, order, forder)

    # If the graph is dinconnected, connect it with trivial indices that
    # will be contracted at the very end.
    connect_graph(AA, v, order)  #checked
    while len(order) > 0:
        tcon = get_tcon(v, order[0]) # tcon = tensors to be contracted
        # Find the indices icon that are to be contracted.
        if len(tcon)==1:
            tracing = True
            icon = [order[0]]
        else:
            tracing = False
            icon = get_icon(v, tcon)
        # Position in tcon[0] and tcon[1] of indices to be contracted.
        # In the case of trace, pos2 = []
        pos1, pos2 = get_pos(v, tcon, icon) 
        if tracing:
            pass
            # Trace on a tensor
            #new_A = trace(AA[tcon[0]], axis1=pos1[0], axis2=pos1[1])
        else:
            # Contraction of 2 tensors
            new_A = con(AA[tcon[0]], AA[tcon[1]], (pos1, pos2))
        AA.append(new_A)
        v.append(find_newv(v, tcon, icon))  # Add the v for the new tensor
        for i in sorted(tcon, reverse=True):
            # Delete the contracted tensors and indices from the lists.
            # tcon is reverse sorted so that tensors are removed starting from
            # the end of AA, otherwise the order would get messed.
            del AA[i]
            del v[i]
        order = renew_order(order, icon)  # Update order

    vlast = v[0]
    A = AA[0]
    A = permute_final(A, vlast, forder)
    return A

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def create_order(v):
    """ Identify all unique, positive indices and return them sorted. """
    flat_v = sum(v, [])
    x = [i for i in flat_v if i>0]
    # Converting to a set and back removes duplicates
    x = list(set(x))
    return sorted(x)


def create_forder(v):
    """ Identify all unique, negative indices and return them reverse sorted
    (-1 first).
    """
    flat_v = sum(v, [])
    x = [i for i in flat_v if i<0]
    # Converting to a set and back removes duplicates
    x = list(set(x))
    return sorted(x, reverse=True)


def connect_graph(AA, v, order):
    """ Connect the graph of tensors to be contracted by trivial
    indices, if necessary. Add these trivial indices to the end of the
    contraction order.

    AA, v and order are modified in place.
    """
    # Build ccomponents, a list of the connected components of the graph,
    # where each component is represented by a a set of indices.
    unvisited = set(range(len(AA)))
    visited = set()
    ccomponents = []
    while unvisited:
        component = set()
        next_visit = unvisited.pop()
        to_visit = {next_visit}
        while to_visit:
            i = to_visit.pop()
            unvisited.discard(i)
            component.add(i)
            visited.add(i)
            # Get the indices of tensors neighbouring AA[i].
            i_inds = set(v[i])
            neighs = (j for j, j_inds in enumerate(v) if i_inds.intersection(j_inds))
            for neigh in neighs:
                if neigh not in visited:
                    to_visit.add(neigh)
        ccomponents.append(component)
    # If there is more than one connected component, take one of them, a
    # take an arbitrary tensor (called c) out of it, and connect that
    # tensor with an arbitrary tensor (called d) from all the other
    # components using a trivial index.
    c = ccomponents.pop().pop()
    while ccomponents:
        d = ccomponents.pop().pop()
        A_c = AA[c]
        A_d = AA[d]
        c_axis = len(v[c])
        d_axis = len(v[d])
        AA[c] = tf.expand_dims(A_c, c_axis)
        AA[d] = tf.expand_dims(A_d, d_axis)
        try:
            dim_num = max(order)+1
        except ValueError:
            dim_num = 1
        v[c].append(dim_num)
        v[d].append(dim_num)
        order.append(dim_num)
    return None


def get_tcon(v, index):
    """ Gets the list indices in AA of the tensors that have index as their
    leg.
    """
    tcon = []
    for i, inds in enumerate(v):
        if index in inds:
            tcon.append(i)
    l = len(tcon)
    # If check_indices is called and it does its work properly then these
    # checks should in fact be unnecessary.
    if l > 2:
        raise ValueError('In ncon.get_tcon, more than two tensors share a '
                'contraction index.')
    elif l < 1:
        raise ValueError('In ncon.get_tcon, less than one tensor share a '
                'contraction index.')
    elif l == 1:
        # The contraction is a trace.
        how_many = v[tcon[0]].count(index)
        if how_many != 2:
            # Only one tensor has this index but it is not a trace because it
            # does not occur twice for that tensor.
            raise ValueError('In ncon.get_tcon, a trace index is listed '
                    '!= 2 times for the same tensor.')
    return tcon


def get_icon(v, tcon):
    """ Returns a list of indices that are to be contracted when contractions
    between the two tensors numbered in tcon are contracted. """
    inds1 = v[tcon[0]]
    inds2 = v[tcon[1]]
    icon = set(inds1).intersection(inds2)
    icon = list(icon)
    return icon


def get_pos(v, tcon, icon):
    """ Get the positions of the indices icon in the list of legs the tensors
    tcon to be contracted.
    """
    pos1 = [[i for i, x in enumerate(v[tcon[0]]) if x == e] for e in icon]
    pos1 = sum(pos1, [])
    if len(tcon) < 2:
        pos2 = []
    else:
        pos2 = [[i for i, x in enumerate(v[tcon[1]]) if x == e] for e in icon]
        pos2 = sum(pos2, [])
    return pos1, pos2


def find_newv(v, tcon, icon):
    """ Find the list of indices for the new tensor after contraction of
    indices icon of the tensors tcon.
    """
    if len(tcon) == 2:
        newv = v[tcon[0]] + v[tcon[1]]
    else:
        newv = v[tcon[0]]
    newv = [i for i in newv if i not in icon]
    return newv


def renew_order(order, icon):
    """ Returns the new order with the contracted indices removed from it. """
    return [i for i in order if i not in icon]


def permute_final(A, v, forder):
    """ Returns the final tensor A with its legs permuted to the order given
    in forder.
    """
    perm = [v.index(i) for i in forder]
    permuted = tf.transpose(A, tuple(perm))
    return permuted


def do_check_indices(AA, v, order, forder):
    """ Check that
    1) the number of tensors in AA matches the number of index lists in v.
    2) every tensor is given the right number of indices.
    3) every contracted index is featured exactly twice and every free index
       exactly once.
    4) the dimensions of the two ends of each contracted index match.
    """
    
    #1)
    if len(AA) != len(v):
        raise ValueError(('In ncon.do_check_indices, the number of tensors %i'
                            ' does not match the number of index lists %i')
                           %(len(AA), len(v)))

    #2)
    # Create a list of lists with the shapes of each A in AA.
    shapes = list(map(lambda A: list(A.get_shape()), AA))
    for i,inds in enumerate(v):
        if len(inds) != len(shapes[i]):
            raise ValueError(('In ncon.do_check_indices, len(v[%i])=%i '
                                'does not match the numbers of indices of '
                                'AA[%i] = %i')%(i, len(inds), i,
                                                len(shapes[i])))

    #3) and 4)
    # v_pairs = [[(0,0), (0,1), (0,2), ...], [(1,0), (1,1), (1,2), ...], ...]
    v_pairs  = [[(i,j) for j in range(len(s))] for i, s in enumerate(v)]
    v_pairs = sum(v_pairs, [])
    v_sum = sum(v, [])
    # For t, o in zip(v_pairs, v_sum) t is the tuple of the number of
    # the tensor and the index and o is the contraction order of that
    # index. We group these tuples by the contraction order.
    order_groups = [[t for t, o in zip(v_pairs, v_sum) if o == e]
                    for e in order]
    forder_groups = [[1 for fo in v_sum if fo == e] for e in forder]
    for i, o in enumerate(order_groups):
        if len(o) != 2:
            raise ValueError(('In ncon.do_check_indices, the contracted index '
                    '%i is not featured exactly twice in v.')%order[i])
        else:
            A0, ind0 = o[0]
            A1, ind1 = o[1]
            compatible = AA[A0].get_shape()[ind0] == AA[A1].get_shape()[ind1]
            if not compatible: 
                raise ValueError('In ncon.do_check_indices, for the '
                                 'contraction index %i, the leg %i of tensor '
                                 'number %i and the leg %i of tensor number '
                                 '%i are not compatible.'
                                 %(order[i], ind0, A0, ind1, A1))
    for i, fo in enumerate(forder_groups):
        if len(fo) != 1:
            raise ValueError(('In ncon.do_check_indices, the free index '
                    '%i is not featured exactly once in v.')%forder[i])

    # All is well if we made it here.
    return True



def con(A, B, inds):
    return tf.tensordot(A, B, inds)



def trace(A, axis1=0, axis2=1):
    return A.trace(axis1=axis1, axis2=axis2)

