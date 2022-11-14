"""Universal functions which work on both torch tensors and numpy arrays."""
import builtins
import collections
from types import GeneratorType

# TODO: check what is needed for yolo
import numpy as np
import torch
import torch.nn.functional as F


class Type:
    def __init__(self, as_numpy, as_torch):
        self.as_numpy = as_numpy
        self.as_torch = as_torch

    def __str__(self):
        return "U" + str(self.as_numpy)

    @property
    def sizeof(self):
        """Returns the size of this type, in bytes."""
        return np.dtype(self.as_numpy).itemsize


float16 = Type(np.float16, torch.float16)
float32 = Type(np.float32, torch.float32)
float64 = Type(np.float64, torch.float64)

uint8 = Type(np.uint8, torch.uint8)
int16 = Type(np.int16, torch.int16)
int32 = Type(np.int32, torch.int32)
int64 = Type(np.int64, torch.int64)

torch_has_bool = hasattr(torch, "bool")
torch_bool = getattr(torch, "bool", torch.uint8)
# don't rename bool_type to bool, avoid collision with builtins.bool
bool_type = Type(np.bool, torch_bool)

types = (float16, float32, float64, uint8, int16, int32, int64, bool_type)
float_types = (float16, float32, float64)
int_types = (uint8, int16, int32, int64)  # excludes bool_type!

_torch_to_type = {v.as_torch: v for v in types if v != bool_type or torch_has_bool}

primitive_types = (int, float, bool)
numpy_scalar_types = tuple(t.as_numpy for t in types)

scalar_int_types = (int, np.uint8, np.int16, np.int32, np.int64)
scalar_float_types = (float, np.float16, np.float32, np.float64)
scalar_numeric_types = scalar_int_types + scalar_float_types
all_scalar_types = primitive_types + numpy_scalar_types


def get_type(x):
    """Returns an element of U.types, e.g. U.uint8 or U.float32."""
    if torch.is_tensor(x):
        return _torch_to_type[x.dtype]
    else:
        # make it work for primitive types and lists
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        # OK, so _np_to_type[t.dtype] does not work.
        # For example, np.uint8 is supposed to be equal to dtype("uint8"), but when trying to index the dict,
        # suddenly dtype("uint8") is not found.
        for typ in types:
            if typ.as_numpy == x.dtype:
                return typ

        raise KeyError("%s is a numpy dtype unsupported by framework.utils.universal" % x.dtype)


def unary(op_name, array):
    if torch.is_tensor(array):
        return getattr(torch, op_name)(array)
    else:
        return getattr(np, op_name)(array)


def abs(array):
    return unary("abs", array)


def sqrt(array):
    return unary("sqrt", array)


def log(array):
    return unary("log", array)


def exp(array):
    return unary("exp", array)


def floor(array):
    return unary("floor", array)


def ceil(array):
    return unary("ceil", array)


def reduce(torch_reduce, np_reduce, array, axis=None, keepdims=False):
    """Also works for multiple axis specified at once, but these axes must form a contiguous range."""
    if torch.is_tensor(array):
        s = shape(array)
        n = len(s)

        if n == 0:  # scalar torch tensor
            assert not axis, "Axis should not be specified for scalars"
            return np_reduce([array.item()])

        final_shape = None  # in case reshape is needed at the end

        if axis is None:
            axis = tuple(range(n))  # all axes

        if isinstance(axis, (list, tuple)):  # multiple axes
            assert axis, "Cannot reduce over no axes (axis parameter is an empty list or tuple)"
            if len(axis) == 1:
                axis = axis[0]
            else:
                # 2 or more axes
                axis = sorted(set(a if a >= 0 else n + a for a in axis))  # "normalize" axis indices
                assert axis[-1] - axis[0] == len(axis) - 1, \
                    "Reduction axes must form a contiguous range, %s is not supported" % (axis,)
                # reshape the tensor and reduce the axis list to only one axis
                new_shape = s[:axis[0]] + (-1,) + s[axis[-1] + 1:]
                if keepdims:
                    final_shape = s[:axis[0]] + (1,) * len(axis) + s[axis[-1] + 1:]  # re-expand the dims
                # print(s, axis, new_shape, final_shape, axis[0])
                array = array.contiguous().view(*new_shape)
                axis = axis[0]

        result = torch_reduce(array, dim=axis, keepdim=keepdims)
        if isinstance(result, tuple):  # min and max also return the indices
            assert len(result) == 2
            result = result[0]
        if final_shape is not None:
            result = result.view(*final_shape)  # no need for .contiguous, just expanding dims
        return result
    else:
        return np_reduce(array, axis=axis, keepdims=keepdims)


def sum(array, axis=None, keepdims=False):
    return reduce(torch.sum, np.sum, array, axis=axis, keepdims=keepdims)


def mean(array, axis=None, keepdims=False):
    return reduce(torch.mean, np.mean, array, axis=axis, keepdims=keepdims)


def median(array):
    if torch.is_tensor(array):
        return torch.median(array)
    else:
        return np.median(array)


def histogram(array, bins: int, range: tuple):
    assert isinstance(bins, int), "'bins' must be an int"
    rmin, rmax = range
    assert isinstance(rmin, float) and isinstance(rmax, float), "'range' must be a 2-tuple of floats"

    if torch.is_tensor(array):
        return torch.histc(array, bins=bins, min=rmin, max=rmax)
    else:
        return np.histogram(array, bins=bins, range=(rmin, rmax))[0]


def cumsum(array, axis=None, dtype=None):
    if torch.is_tensor(array):
        return torch.cumsum(array, dim=axis, dtype=None if dtype is None else dtype.as_torch)
    else:
        return np.cumsum(array, axis=axis, dtype=None if dtype is None else dtype.as_numpy)


def std(array, axis=None, keepdims=False):
    return reduce(torch.std, np.std, array, axis=axis, keepdims=keepdims)


def amin(array, axis=None, keepdims=False):
    return reduce(torch.min, np.amin, array, axis=axis, keepdims=keepdims)


def amax(array, axis=None, keepdims=False):
    return reduce(torch.max, np.amax, array, axis=axis, keepdims=keepdims)


def lerp(start, end, weight):
    if torch.is_tensor(start) or torch.is_tensor(end):
        start, end = to_tensors(start, end)
        return torch.lerp(start, end, weight)
    else:
        return start + (end - start) * weight


def isin(array, positive_set):
    """Returns a boolean mask with the same shape as 'a', which is True where the corresponding element is in the given set.

    Args:
        array: the input; tensor, ndarray or primitive type.
        positive_set: primitive type, list, tuple or numpy array. The values against which to test each element of a.
    Returns:
        a boolean mask with the same shape as 'a', which is True where the corresponding element is in the given set.
    """
    if isinstance(positive_set, primitive_types):
        positive_set = [positive_set]
    assert isinstance(positive_set, (list, tuple, np.ndarray))

    if torch.is_tensor(array):
        if isinstance(positive_set, np.ndarray):
            positive_set = positive_set.flatten()

        mask = None
        for i in positive_set:
            index_mask = (array == i)
            if mask is None:
                mask = index_mask
            else:
                mask = mask | index_mask

        if mask is None:
            return torch.zeros_like(array, dtype=torch_bool)
        return mask

    elif isinstance(array, np.ndarray):
        return np.isin(array, positive_set)

    elif isinstance(array, primitive_types):
        if isinstance(positive_set, np.ndarray):
            positive_set = positive_set.tolist()
        return array in positive_set

    else:
        raise TypeError("Invalid type of array: %s" % type(array))


def is_namedtuple(x):
    return isinstance(x, tuple) and hasattr(x, "_fields")


def apply_recursively(data, transform, filter=None, keep_namedtuples: bool = True, apply_on_struct: bool = False):
    """
    Apply a transform on all elements of the data.

    Recurses into the following data structures:
        * collections.Mapping, e.g. dict;
        * collections.Sequence (except strings); e.g. list/tuple;
        * collections.Set, e.g. set.

    Args:
        data: anything: a dict, list, or simple data type.
        transform: a function to be applied to all elements.
        filter: if not None, must be a callable returning a bool,
            that determines whether the transform should be applied to an element (True) or not (False).
            Will be called e.g. on elements of list/tuples, and on values of dicts.
        keep_namedtuples: if False, namedtuples are converted to regular tuples.
            Useful if your namedtuples require typed arguments and "transform" changes the argument types.
        apply_on_struct: if True, the transform will also be applied to the data structures (list, dict, ...) themselves.

    Returns:
        the same structure as the input data, but all items for which filter returned True are transformed by "transform".
        The result is always a completely new object.
    """
    assert isinstance(keep_namedtuples, bool)
    assert isinstance(apply_on_struct, bool)

    def recurse_into(v):
        return apply_recursively(v, transform, filter, keep_namedtuples, apply_on_struct=apply_on_struct)

    if isinstance(data, collections.Mapping):
        # Be careful to keep the order of OrderedDict here!
        # Making an intermediate dict ({}) would ruin this.
        result = type(data)()
        for k, v in data.items():
            result[k] = recurse_into(v)

        if apply_on_struct and (filter is None or filter(data)):
            result = transform(result)
        return result

    elif is_namedtuple(data):
        if keep_namedtuples:
            result = type(data)(*[recurse_into(l) for l in data])
        else:
            result = tuple(recurse_into(l) for l in data)

        if apply_on_struct and (filter is None or filter(data)):
            result = transform(result)
        return result

    elif isinstance(data, (collections.Sequence, collections.Set)) and not isinstance(data, str):
        result = type(data)(recurse_into(l) for l in data)

        if apply_on_struct and (filter is None or filter(data)):
            result = transform(result)
        return result

    elif filter is None or filter(data):
        return transform(data)

    else:
        return data


def index_select(array, indices, axis=0):
    """Equivalent to array[:,:,:,indices] in numpy (the number of colons being axis-1),
    and a better version of index_select in torch.
    """
    if axis < 0:
        axis += ndim(array)
    if torch.is_tensor(array):
        indices = cast_like(indices, array, cast_dtype=False)

        # Unfortunately array[indices] does not work in torch yet,
        # and index_select only works for 1D indices, so we work it around.

        if indices.dim() == 0:
            return array[(slice(None),) * (axis - 1) + (indices,)]
        else:
            indices_shape = shape(indices)
            if len(indices_shape) > 1:
                indices = flatten(indices)
            result = torch.index_select(array, dim=axis, index=indices.long())
            if len(indices_shape) > 1:
                result_shape = shape(result)
                result = reshape(result, result_shape[:axis] + indices_shape + result_shape[axis + 1:])
            return result
    else:
        indices = to_numpy(indices)
        return np.take(array, indices, axis=axis)


def masked_select(array, mask):
    """Bool mask select operation."""
    if torch.is_tensor(array):
        mask = cast_like(mask, array, cast_dtype=False)
        return torch.masked_select(array, mask)
    else:
        mask = to_numpy(mask)
        return array[mask]


def gather(_sentinel=None, array=None, indices=None, axis: int = None):
    """Essentially batched array indexing.
    array and indices must have the same dimensions apart from "axis".
    These other dimensions should be regarded as batch dimensions.

    In the remaining dimension, "indices" represents a list of integer indices (i),
    and "array" contains the array which should be indexed using these indices (a).
    The result will be: [a[x] for x in i].
    Now do this in batch and you obtain the gather operation.

    Args:
        array: array to index, ndarray or tensor.
        indices: indices to use, ndarray or tensor.
        axis: the axis where indexing should occur.

    Returns:
        an array with the same shape as indices, containing the corresponding values from the source array.
    """
    assert _sentinel is None, "Use keyword arguments for U.gather: " \
                              "torch.gather has weird parameter order which differs from np.take_along_axis"
    assert array is not None
    assert indices is not None
    assert axis is not None

    if torch.is_tensor(array):
        indices = cast_like(indices, array, cast_dtype=False)
        return torch.gather(self=array, dim=axis, index=indices)
    else:
        indices = to_numpy(indices)
        return np.take_along_axis(arr=array, indices=indices, axis=axis)


def swap_axes(array, i, j):
    if torch.is_tensor(array):
        return torch.transpose(array, i, j)
    else:
        return np.swapaxes(array, i, j)


def unique(array, return_counts: bool = False):
    if torch.is_tensor(array):
        assert get_type(array) in int_types + (
        bool_type,), "unique can only be called on int or bool inputs, not %s" % (
            get_type(array),)
        if return_counts:
            array = flatten(array)  # torch.flatten returns 0-dim tensor for 0-dim tensors...
            assert array.dim() == 1
            n = array.size(0)
            if n <= 1:
                return array, torch.ones(size=[n], dtype=torch.int64, device=array.device)

            array, _ = torch.sort(array)
            # 1: last element of a same-value sequence, 0: elsewhere
            single_true = torch.ones(size=[1], dtype=torch.bool, device=array.device)
            # TODO the to(torch.bool) can be removed after everyone has torch 1.2
            mask = torch.cat(((array[:-1] != array[1:]).to(torch.bool), single_true))
            values = torch.masked_select(array, mask)
            run_ends = flatten(torch.nonzero(mask)) + 1
            run_starts = torch.cat((torch.zeros(size=[1], dtype=run_ends.dtype, device=run_ends.device), run_ends[:-1]))
            counts = run_ends - run_starts
            return values, counts
        else:
            return torch.unique(array)
    elif isinstance(array, primitive_types):
        if return_counts:
            return [array], [1]
        else:
            return [array]
    else:
        return np.unique(array, return_counts=return_counts)


def argmax(a, axis, keepdims=False):
    assert isinstance(axis, int)
    if axis < 0:
        axis += ndim(a)

    if torch.is_tensor(a):
        return torch.argmax(a, dim=axis, keepdim=keepdims)
    else:
        a = np.argmax(a, axis=axis)
        if keepdims:
            a = np.expand_dims(a, axis)
        return a


def argmin(a, axis, keepdims=False):
    assert isinstance(axis, int)
    if axis < 0:
        axis += ndim(a)

    if torch.is_tensor(a):
        return torch.argmin(a, dim=axis, keepdim=keepdims)
    else:
        a = np.argmin(a, axis=axis)
        if keepdims:
            a = np.expand_dims(a, axis)
        return a


def argsort(a, axis):
    if torch.is_tensor(a):
        return torch.argsort(a, axis)
    else:
        return np.argsort(a, axis)


def minimum(a, b):
    if torch.is_tensor(a) or torch.is_tensor(b):
        a, b = to_tensors(a, b)
        return torch.min(a, b)
    else:
        return np.minimum(a, b)


def maximum(a, b):
    if torch.is_tensor(a) or torch.is_tensor(b):
        a, b = to_tensors(a, b)
        return torch.max(a, b)
    else:
        return np.maximum(a, b)


def relu(a):
    if torch.is_tensor(a):
        return F.relu(a)
    else:
        return np.maximum(0, a)


def round(array, dtype=None):
    """Rounds array to the nearest integer.

    Args:
        array: tensor or ndarray.
        dtype: optional. Destination dtype, e.g. U.uint8.
            If not given, the array's type is not changed (in particular, if the array was float it will remain float).

    Returns:
        the rounded and optionally dtype-casted array.
    """
    if get_type(array) in float_types:
        if torch.is_tensor(array):
            array = torch.round(array)
        else:
            array = np.around(array)

    if dtype is not None:
        assert dtype in int_types, dtype
        array = cast(array, dtype)

    return array


def isnan(array):
    if torch.is_tensor(array):
        if get_type(array) in int_types + (bool_type,):
            return zeros_like(array, dtype=bool_type)
        else:
            return torch.isnan(array)
    else:
        return np.isnan(array)


def isfinite(array):
    if torch.is_tensor(array):
        if get_type(array) in int_types + (bool_type,):
            return ones_like(array, dtype=bool_type)
        else:
            return torch.isfinite(array)
    else:
        return np.isfinite(array)


def all(array):
    if isinstance(array, (GeneratorType, list, tuple)):
        return builtins.all(array)
    elif torch.is_tensor(array):
        return array.all()
    else:
        return np.all(array)


def any(array):
    if isinstance(array, (GeneratorType, list, tuple)):
        return builtins.any(array)
    elif torch.is_tensor(array):
        return array.any()
    else:
        return np.any(array)


def where(condition, a, b):
    if torch.is_tensor(condition) or torch.is_tensor(a) or torch.is_tensor(b):
        condition, a, b = to_tensors(condition, a, b)
        return torch.where(condition, a, b)
    else:
        return np.where(condition, a, b)


def ndim(array):
    """
    Get number of dimensions for an array.

    Args:
        array: input array.

    Returns:
         an int corresponding to the ndim of array.
    """
    if torch.is_tensor(array):
        return array.dim()
    elif isinstance(array, primitive_types):
        return 0
    elif isinstance(array, (list, tuple)):
        if len(array) == 0:
            return 1
        else:
            return 1 + ndim(array[0])
    elif isinstance(array, np.ndarray):
        return array.ndim
    elif isinstance(array, numpy_scalar_types):
        return 0
    else:
        raise TypeError("Invalid input type %s for ndim()" % (type(array),))


def shape(array):
    """
    Get shape of the array.

    Args:
        array: input array

    Returns:
        the shape of the array as a tuple.
    """
    if torch.is_tensor(array):
        return tuple(map(int, array.size()))
    elif isinstance(array, primitive_types):
        return ()
    elif isinstance(array, (list, tuple)):
        if len(array) == 0:
            return (0,)
        else:
            return (len(array),) + shape(array[0])
    elif isinstance(array, np.ndarray):
        return array.shape
    elif isinstance(array, numpy_scalar_types):
        return ()
    else:
        raise TypeError("Invalid input type %s for shape()" % (type(array),))


def volume(array):
    """Gets the total number of elements in an array.
    Named "volume" to avoid a bit of confusion about "size".
    """
    if torch.is_tensor(array):
        return array.numel()
    elif isinstance(array, primitive_types):
        return 1
    elif isinstance(array, (list, tuple)):
        return sum(volume(x) for x in array)
    elif isinstance(array, numpy_scalar_types):
        return 1
    else:
        return array.size


def item(x):
    """Takes a one-element array and returns its only value."""
    assert volume(x) == 1, "item() only works on a one-element array, but this one has %s elements" % volume(x)
    if torch.is_tensor(x):
        return x.item()
    elif isinstance(x, primitive_types):
        return x
    elif isinstance(x, (list, tuple)):
        return item(x[0])
    else:
        return np.asarray(x).flatten()[0]


def _new_like(array, kind, shape, dtype, **kwargs):
    assert dtype is None or isinstance(dtype, Type), "dtype must be None or an instance of U.Type, e.g. U.uint8"

    if torch.is_tensor(array):
        dtype = dtype.as_torch if dtype is not None else array.dtype
        if shape is not None:
            return getattr(array, "new_" + kind)(shape, dtype=dtype, **kwargs)
        else:
            return getattr(torch, kind + "_like")(array, dtype=dtype, **kwargs)
    else:
        assert isinstance(array, np.ndarray)
        dtype = dtype.as_numpy if dtype is not None else array.dtype
        if shape is not None:
            return getattr(np, kind)(shape=shape, dtype=dtype, **kwargs)
        else:
            return getattr(np, kind + "_like")(array, dtype=dtype, **kwargs)


def zeros_like(array, shape=None, dtype=None):
    return _new_like(array, "zeros", shape, dtype)


def ones_like(array, shape=None, dtype=None):
    return _new_like(array, "ones", shape, dtype)


def empty_like(array, shape=None, dtype=None):
    return _new_like(array, "empty", shape, dtype)


def full_like(array, fill_value, shape=None, dtype=None):
    return _new_like(array, "full", shape, dtype, fill_value=fill_value)


def concatenate(arrays, axis=0):
    if any(torch.is_tensor(a) for a in arrays):
        arrays = to_tensors(*arrays)
        return torch.cat(arrays, dim=axis)
    else:
        return np.concatenate(arrays, axis=axis)


def split(array, num_or_size_splits, axis):
    axis_size = shape(array)[axis]

    if isinstance(num_or_size_splits, int):
        # = number of splits
        assert axis_size % num_or_size_splits == 0, \
            "Size of axis %s (%s) not divisible by number of splits (%s)" % (axis, axis_size, num_or_size_splits)
    else:
        # = sizes of splits
        s = sum(num_or_size_splits)
        assert s == axis_size, \
            "Sum of split sizes (%s = sum(%s)) does not equal size of axis %s (%s)" % (
                s, num_or_size_splits, axis, axis_size)

    if torch.is_tensor(array):
        if isinstance(num_or_size_splits, int):
            split_size_or_sections = axis_size // num_or_size_splits
        else:
            split_size_or_sections = num_or_size_splits
        return torch.split(array, split_size_or_sections=split_size_or_sections, dim=axis)
    else:
        if isinstance(num_or_size_splits, int):
            num_or_index = num_or_size_splits
        else:
            num_or_index = np.cumsum(num_or_size_splits)[:-1]
        return np.split(array, indices_or_sections=num_or_index, axis=axis)


def stack(arrays, axis=0):
    if any(torch.is_tensor(a) for a in arrays):
        arrays = to_tensors(*arrays)
        return torch.stack(arrays, dim=axis)
    else:
        return np.stack(arrays, axis=axis)


def unstack(array, axis=0):
    if torch.is_tensor(array):
        return torch.unbind(array, dim=axis)
    else:
        parts = np.split(array, array.shape[axis], axis)
        return tuple(np.squeeze(part, axis) for part in parts)


def tile(array, reps):
    if torch.is_tensor(array):
        return array.repeat(*reps)
    else:
        return np.tile(array, reps)


def square(array):
    if torch.is_tensor(array):
        return array * array
    else:
        return np.square(array)


def inverse(matrix):
    if torch.is_tensor(matrix):
        return torch.inverse(matrix)
    else:
        return np.linalg.inv(matrix)


def batch_inverse(array):
    if torch.is_tensor(array):
        assert ndim(array) == 3, shape(array)
        return torch.stack([m.inverse() for m in torch.unbind(array)])
    else:
        return np.linalg.inv(array)  # already works for batches


def batch_matmul(a, b):
    # Very convenient: both torch and numpy already work for batches!
    return a @ b


def clip(a, a_min, a_max):
    if torch.is_tensor(a):
        # a, a_min, a_max = to_tensors(a, a_min, a_max) # not needed!
        return torch.clamp(a, a_min, a_max)
    else:
        return np.clip(a, a_min, a_max)


def flip(a, axis):
    assert isinstance(axis, int), axis
    if torch.is_tensor(a):
        if axis < 0:
            axis += ndim(a)
        return torch.flip(a, [axis])
    else:
        return np.flip(a, axis=axis)


def reshape(a, new_shape):
    if isinstance(new_shape, int):
        new_shape = (new_shape,)
    elif isinstance(new_shape, list):
        new_shape = tuple(new_shape)

    if 0 in new_shape:
        s = shape(a)
        new_shape = tuple([s[i] if new_shape[i] == 0 else new_shape[i] for i in range(len(new_shape))])

    if torch.is_tensor(a):
        try:
            return a.contiguous().view(new_shape)
        except RuntimeError as e:
            # torch's error message is not informative enough
            raise RuntimeError(
                "RESHAPE ERROR: Original %s (volume=%s, dtype=%s), new shape would be %s (volume=%s), error: %s" %
                (shape(a), volume(a), get_type(a), new_shape, np.prod(new_shape), e))
    else:
        return np.reshape(a, new_shape)


def flatten(a):
    """Use this also if you are sure that the input is a torch tensor!
    torch.flatten is buggy, returns 0-dim tensor for 0-dim tensors..."""
    return reshape(a, -1)


def expand_dims(a, axis):
    if torch.is_tensor(a):
        return torch.unsqueeze(a, axis)
    else:
        return np.expand_dims(a, axis)


def squeeze(a, axis=None):
    if torch.is_tensor(a):
        assert axis is None or isinstance(axis, int), "Squeezing at axis=%s not supported" % (axis,)
        if axis is None:
            return torch.squeeze(a)
        else:
            return torch.squeeze(a, axis)
    else:
        return np.squeeze(a, axis)


def left_broadcast(input, template):
    """Broadcast operation when the input is aligned to the template on the left.
    If the input has fewer dimensions than the template, it is expanded with extra dimensions at the right.
    (The usual broadcast extends dimensions on the left.)

    Returns:
        input, with extra dimensions added at the end.
    """
    input_shape = shape(input)
    input_dim = len(input_shape)
    template_dim = ndim(template)
    assert input_dim <= template_dim, \
        "Input cannot have more dimensions than template when broadcasting (%s vs %s)" % (input_dim, template_dim)
    if input_dim < template_dim:
        input = reshape(input, input_shape + (1,) * (template_dim - input_dim))
    return input


def cast(a, new_type: Type):
    assert isinstance(new_type, Type)
    if torch.is_tensor(a):
        return a.to(dtype=new_type.as_torch)
    elif isinstance(a, primitive_types):
        if new_type in float_types:
            return float(a)
        elif new_type in int_types:
            return int(a)
        elif new_type == bool_type:
            return bool(a)
        else:
            raise TypeError("Invalid target type %s" % (new_type,))
    elif isinstance(a, (list, tuple)):
        return np.asarray(a, new_type.as_numpy)
    elif isinstance(a, np.ndarray):
        return a.astype(new_type.as_numpy)
    else:
        raise TypeError("Unknown input type %s for cast()" % (type(a),))


def to_float(a):
    """Promotes the input to float.
    If it is already a float, keeps it as it is.
    Otherwise it will be converted to float32 or float64, depending on whether float32 can represent it."""
    t = get_type(a)
    if t in float_types:
        return a
    elif t in [bool_type, uint8, int16]:
        return cast(a, float32)
    elif t in [int32, int64]:
        return cast(a, float64)
    else:
        raise NotImplementedError("Please implement casting %s to float" % (t,))


def pow(a, exponent):
    if torch.is_tensor(a) or torch.is_tensor(exponent):
        a, exponent = to_tensors(a, exponent)
        return a.pow(exponent)
    else:
        return np.power(a, exponent)


def _unary(numpy_op, torch_op):
    def _do_op(a):
        op = torch_op if torch.is_tensor(a) else numpy_op
        return op(a)

    return _do_op


sin = _unary(np.sin, torch.sin)
cos = _unary(np.cos, torch.cos)
tan = _unary(np.tan, torch.tan)
asin = _unary(np.arcsin, torch.asin)
acos = _unary(np.arccos, torch.acos)
atan = _unary(np.arctan, torch.atan)
sinh = _unary(np.sinh, torch.sinh)
cosh = _unary(np.cosh, torch.cosh)
tanh = _unary(np.tanh, torch.tanh)
log1p = _unary(np.log1p, torch.log1p)


def atan2(y, x):
    if torch.is_tensor(y) or torch.is_tensor(x):
        return torch.atan2(y, x)
    else:
        return np.arctan2(y, x)


def atanh(y):
    if torch.is_tensor(y):
        # Numerically stable, because the composing operations have no common singularities.
        # (the singularities are at y=-1 and y=1, respectively)
        return 0.5 * (torch.log1p(y) - torch.log1p(-y))
    else:
        return np.arctanh(y)


def sigmoid(x):
    if torch.is_tensor(x):
        return torch.sigmoid(x)
    else:
        return 1 / (1 + np.exp(-x))


def logsigmoid(x):
    if torch.is_tensor(x):
        return F.logsigmoid(x)
    else:
        return -np.log1p(np.exp(-x))


def random_normal_like(a, mean, std, rng=None):
    if torch.is_tensor(a):
        r = torch.empty_like(a)
        r.normal_(mean=mean, std=std)
        return r
    else:
        if rng is None:
            rng = np.random
        return rng.normal(loc=mean, scale=std, size=shape(a))


def get_device(t):
    """Gets the device of a torch tensor, a numpy array or a primitive type."""
    if torch.is_tensor(t):
        return t.device
    else:
        return torch.device("cpu")


def ascontiguousarray(t):
    if torch.is_tensor(t):
        return t.contiguous()
    else:
        # unfortunately np.ascontiguousarray is buggy: it can leave <0 strides for dimensions equal to 1
        # but np.copy is not
        if any(x < 0 for x in t.strides):
            return np.copy(t)
        else:
            return np.ascontiguousarray(t)


def cast_like(t, template, cast_dtype=True):
    """Casts a numpy array or tensor so it will have the same
        * dtype,
        * tensorness (torch.is_tensor)
        * and device
        as the other argument.

    Args:
        t: the tensor to cast.
        template: t will be casted to the type of this object.
        cast_dtype: if False, dtype is not casted, only the device and tensorness are changed
            to make t compatible with template when doing e.g. binary operations.

    Returns:
        the casted array/tensor, having the same properties as the template.
    """

    # if get_device(t) != get_device(template):
    #    print("Warning: inter-device movement: %s bytes (%s) from %s to %s" %
    #          (volume(t)*get_type(t).sizeof, shape(t), get_device(t), get_device(template)))

    if torch.is_tensor(template):
        if not torch.is_tensor(t):
            if isinstance(t, (list, tuple)):
                t = np.array(t, dtype=get_type(template).as_numpy if cast_dtype else None)
            elif isinstance(t, primitive_types):
                t = np.array(t, dtype=type(t))

            if t.dtype == np.bool and not torch_has_bool:
                t = t.astype(np.uint8)

            if any(x < 0 for x in t.strides):
                # np.ascontiguousarray won't work
                # it does not fix negative strides if the axis has length 1...
                t = np.copy(t)

            t = torch.from_numpy(t)

        t = t.to(device=template.device, dtype=template.dtype if cast_dtype else None, non_blocking=True)
    else:
        if not cast_dtype:
            dtype = None
        elif isinstance(template, int):
            dtype = np.int64
        elif isinstance(template, float):
            dtype = np.float64
        else:
            assert isinstance(template, numpy_scalar_types + (np.ndarray,)), "Cannot cast like %s (type: %s)" % (
                template, type(template))
            dtype = template.dtype

        if isinstance(t, (list, tuple)):
            t = np.array(t, dtype=dtype if cast_dtype else None)
        else:
            if torch.is_tensor(t):
                t = to_numpy(t)
            elif isinstance(t, primitive_types):
                t = np.array(t, dtype=type(t))

            if cast_dtype:
                t = t.astype(dtype)

    return t


def to_numpy(t, detach=True):
    def _to_numpy(t):
        if t is None:
            return None
        elif torch.is_tensor(t):
            t = t.cpu()
            if detach:
                t = t.detach()
            return t.numpy()
        elif hasattr(t, "to_numpy"):
            assert callable(t.to_numpy)
            return t.to_numpy()
        else:
            return t

    return apply_recursively(t, _to_numpy)


def to_tensor(t, device=None):
    """Converts everything to tensor (torch tensor, int, float, bool, ndarray).
    float is converted to FloatTensor, not DoubleTensor.

    Returns:
        a torch.Tensor instance.
    """
    if torch.is_tensor(t):
        if device is not None:
            t = t.to(device=device, non_blocking=True)
        return t

    if isinstance(t, int):
        t = np.asarray(t, dtype=np.int64)
    elif isinstance(t, float):
        t = np.asarray(t, dtype=np.float32)
    elif isinstance(t, bool):
        t = np.asarray(t, dtype=np.uint8)

    assert isinstance(t, np.ndarray), "Cannot convert %s to tensor" % (type(t),)
    if t.dtype == np.bool and not torch_has_bool:
        t = t.astype(np.uint8)

    if any(x < 0 for x in t.strides):
        # np.ascontiguousarray won't work
        # it does not fix negative strides if the axis has length 1...
        t = np.copy(t)

    t = torch.from_numpy(t)

    if device is not None:
        t = t.to(device=device, non_blocking=True)

    return t


def copy(a):
    """Creates a copy/clone of a tensor or ndarray."""
    if torch.is_tensor(a):
        return a.clone()
    elif isinstance(a, primitive_types):
        return a
    elif isinstance(a, (list, tuple)):
        return type(a)(copy(x) for x in a)
    elif isinstance(a, np.ndarray):
        return np.copy(a)
    else:
        raise TypeError("Unknown input type %s for copy()" % (type(a),))


def to_tensors(*arrays):
    """Converts multiple tensors/ndarrays to tensors.
    If at least one of them is a tensor, then the target device is taken from that tensor
    to ensure that all tensors will reside on the same device.
    Ndarrays are converted to tensor, but int and float are left as-is.
    """
    tensors = [x for x in arrays if torch.is_tensor(x)]
    if len(tensors) == len(arrays):
        return arrays
    device = None if not tensors else tensors[0].device
    return [to_tensor(x, device=device) for x in arrays]


def transpose(t, axes):
    assert isinstance(axes, (list, tuple))
    if isinstance(axes, list):
        axes = tuple(axes)

    if torch.is_tensor(t):
        return t.permute(axes)
    else:
        return np.transpose(t, axes)


def pad(t, pad, mode="constant", fill_value=0):
    """
    Pad a tensor in x and/or y dimensions.

    Args:
        t: the tensor or numpy array to pad, shape: ...HW.
        pad: tuple of (pad_left, pad_right, pad_top, pad_bottom)
        mode: "constant", "reflect" or "replicate".
        fill_value: used when mode=="constant", padding value.

    Returns:
        padded tensor
    """
    assert ndim(t) >= 2, \
        ("U.pad expects an at least 2-dimensional input, with (height,width) as the last dimensions, "
         "but it got an input with shape %s") % (shape(t),)
    assert len(pad) == 4, "pad must have 4 values, but is %s" % (pad,)

    if torch.is_tensor(t):
        if mode == "constant":
            return F.pad(t, pad, mode=mode, value=fill_value)
        else:
            return F.pad(t, pad, mode=mode)
    else:
        if mode == "replicate":
            mode = "edge"
        return np.pad(t, ((0, 0),) * (ndim(t) - 2) + (pad[2:], pad[:2]), mode=mode, constant_values=fill_value)


def is_tensor_or_ndarray(t):
    return isinstance(t, np.ndarray) or torch.is_tensor(t)


def squeeze_to_2d(img):
    """Removes batch and/or time and/or channel dimensions from img.

    Works for both numpy and tensorflow.

    Args:
        img: an array with shape [h,w], [1,h,w], [h,w,1], [1,h,w,1], [1,1,h,w] or [1,1,h,w,1]
    Returns:
        img reshaped to [h,w]
    """

    n = ndim(img)
    s = shape(img)
    if n == 5 and s[0] == 1 and s[1] == 1 and s[2] == 1:
        img = img[0, 0, 0, :, :]
    elif n == 4 and s[0] == 1 and s[3] == 1:
        img = img[0, :, :, 0]
    elif n == 4 and s[0] == 1 and s[1] == 1:
        img = img[0, 0, :, :]
    elif n == 3 and s[2] == 1:
        assert s[0] != 1, "squeeze_to_2d ambiguous: %s" % str(s)
        img = img[:, :, 0]
    elif n == 3 and s[0] == 1:
        img = img[0]

    assert ndim(img) == 2, "squeeze_to_2d failed: shape is %s" % str(shape(img))
    return img


def apply_to_tensors(data, transform):
    """
    Apply a transform on all elements of the data that are tensors or numpy arrays.

    Args:
        data: a dict, list, or simple data type
        transform: a function to be applied to all the tensors and numpy arrays

    Returns:
        the same structure as the input data, but all tensors and numpy arrays transformed by "transform"
    """
    return apply_recursively(data, transform, filter=is_tensor_or_ndarray)


def apply_to_nd_tensors(data, transform, n):
    """
    Apply a transform on all elements of the data that are n-dimensional tensors or numpy arrays,
    where n is a fixed integer.

    Args:
        data: a dict, list, or simple data type
        transform: a function to be applied to all the n-dimensional tensors and numpy arrays

    Returns:
        the same structure as the input data, but all n-dim tensors and numpy arrays transformed by "transform"
    """

    def filter(t):
        return is_tensor_or_ndarray(t) and ndim(t) == n

    return apply_recursively(data, transform, filter=filter)


def downsample2x_chw(t, interpolate: bool = True):
    """Downscales a CHW-format tensor or numpy array to half the size.

    Args:
        t: tensor or numpy array, shape=...CHW
        interpolate: bool, whether to use area interpolation (True) or subsampling (False).

    Returns:
        the downsampled tensor, same type as t but half the height and width.
    """
    if interpolate:
        assert get_type(t) in [float32, float64], "Can only downsample2x floating-point images, not %s" % (get_type(t),)
        if torch.is_tensor(t):
            return F.avg_pool2d(t, 2)
        else:
            t1 = t[..., ::2, :] + t[..., 1::2, :]
            t2 = t1[..., ::2] + t1[..., 1::2]
            return t2 / 4
    else:
        return t[..., ::2, ::2]


import unittest


class Tests(unittest.TestCase):
    def test(self):
        # unique
        def test_unique(x):
            gt, counts = np.unique(x, return_counts=True)
            assert np.all(unique(x) == gt)
            pred = np.sort(to_numpy(unique(torch.from_numpy(x))))
            assert np.all(pred == gt), (pred, gt)

            pred, counts_pred = unique(x, return_counts=True)
            assert np.all(pred == gt) and np.all(counts_pred == counts)
            pred, counts_pred = tuple(map(to_numpy, unique(torch.from_numpy(x), return_counts=True)))
            assert np.all(pred == gt) and np.all(counts_pred == counts)

        x = np.random.randint(low=0, high=1000, size=200)
        test_unique(x)
        x = np.random.randint(low=0, high=20, size=1000)
        test_unique(x)
        x = np.random.randint(low=0, high=20, size=1)
        test_unique(x)

        for array in [123, [1, 2, 3], [[2]]]:
            assert ndim(flatten(torch.from_numpy(np.asfarray(array)))) == 1, array

        x = np.random.normal(size=100)
        assert np.amax(np.abs(sigmoid(x) - to_numpy(torch.sigmoid(to_tensor(x))))) < 1e-3, "sigmoid failed"
        gt = to_numpy(F.logsigmoid(to_tensor(x)))
        assert np.amax(np.abs(logsigmoid(x) - gt)) < 1e-3, "logsigmoid failed"
