from functools import wraps, partial
from numbers import Number
from typing import TypeVar, Tuple, Callable

from ...phi import math
from ...phi import geom
from ...phi.geom import Box, Geometry
from ...phi.math import extrapolate_valid_values, DType
from ._field import Field, SampledField
from ._grid import CenteredGrid, Grid, StaggeredGrid
from ._point_cloud import PointCloud
from ._mask import HardGeometryMask


def laplace(field: Grid, axes=None):
    result = field._op1(lambda tensor: math.laplace(tensor, dx=field.dx, padding=field.extrapolation, dims=axes))
    return result


def spatial_gradient(field: CenteredGrid, type: type = CenteredGrid, stack_dim='vector'):
    """
    Finite difference spatial_gradient.

    This function can operate in two modes:

    * `type=CenteredGrid` approximates the spatial_gradient at cell centers using central differences
    * `type=StaggeredGrid` computes the spatial_gradient at face centers of neighbouring cells

    Args:
        field: centered grid of any number of dimensions (scalar field, vector field, tensor field)
        type: either `CenteredGrid` or `StaggeredGrid`
        stack_dim: name of dimension to be added. This dimension lists the spatial_gradient w.r.t. the spatial dimensions.
            The `field` must not have a dimension of the same name.

    Returns:
        spatial_gradient field of type `type`.

    """
    if type == CenteredGrid:
        values = math.gradient(field.values, field.dx.vector.as_channel(name=stack_dim), difference='central', padding=field.extrapolation, stack_dim=stack_dim)
        return CenteredGrid(values, field.bounds, field.extrapolation.spatial_gradient())
    elif type == StaggeredGrid:
        assert stack_dim == 'vector'
        return stagger(field, lambda lower, upper: (upper - lower) / field.dx, field.extrapolation.spatial_gradient())
    raise NotImplementedError(f"{type(field)} not supported. Only CenteredGrid and StaggeredGrid allowed.")


def shift(grid: CenteredGrid, offsets: tuple, stack_dim='shift'):
    """
    Wraps :func:`math.shift` for CenteredGrid.

    Args:
      grid: CenteredGrid: 
      offsets: tuple: 
      stack_dim:  (Default value = 'shift')

    Returns:

    """
    data = math.shift(grid.values, offsets, padding=grid.extrapolation, stack_dim=stack_dim)
    return [CenteredGrid(data[i], grid.box, grid.extrapolation) for i in range(len(offsets))]


def stagger(field: CenteredGrid, face_function: Callable, extrapolation: math.extrapolation.Extrapolation, type: type = StaggeredGrid):
    """
    Creates a new grid by evaluating `face_function` given two neighbouring cells.
    One layer of missing cells is inferred from the extrapolation.
    
    This method returns a Field of type `type` which must be either StaggeredGrid or CenteredGrid.
    When returning a StaggeredGrid, the new values are sampled at the faces of neighbouring cells.
    When returning a CenteredGrid, the new grid has the same resolution as `field`.

    Args:
      field: centered grid
      face_function: function mapping (value1: Tensor, value2: Tensor) -> center_value: Tensor
      extrapolation: extrapolation mode of the returned grid. Has no effect on the values.
      type: one of (StaggeredGrid, CenteredGrid)
      field: CenteredGrid: 
      face_function: Callable:
      extrapolation: math.extrapolation.Extrapolation: 
      type: type:  (Default value = StaggeredGrid)

    Returns:
      grid of type matching the `type` argument

    """
    all_lower = []
    all_upper = []
    if type == StaggeredGrid:
        for dim in field.shape.spatial.names:
            all_upper.append(math.pad(field.values, {dim: (0, 1)}, field.extrapolation))
            all_lower.append(math.pad(field.values, {dim: (1, 0)}, field.extrapolation))
        all_upper = math.channel_stack(all_upper, 'vector')
        all_lower = math.channel_stack(all_lower, 'vector')
        values = face_function(all_lower, all_upper)
        return StaggeredGrid(values, field.bounds, extrapolation)
    elif type == CenteredGrid:
        left, right = math.shift(field.values, (-1, 1), padding=field.extrapolation, stack_dim='vector')
        values = face_function(left, right)
        return CenteredGrid(values, field.bounds, extrapolation)
    else:
        raise ValueError(type)


def divergence(field: Grid) -> CenteredGrid:
    """
    Computes the divergence of a grid using finite differences.

    This function can operate in two modes depending on the type of `field`:

    * `CenteredGrid` approximates the divergence at cell centers using central differences
    * `StaggeredGrid` exactly computes the divergence at cell centers

    Args:
        field: vector field as `CenteredGrid` or `StaggeredGrid`

    Returns:
        Divergence field as `CenteredGrid`
    """
    if isinstance(field, StaggeredGrid):
        components = []
        for i, dim in enumerate(field.shape.spatial.names):
            div_dim = math.gradient(field.values.vector[i], dx=field.dx[i], difference='forward', padding=None, dims=[dim]).spatial_gradient[0]
            components.append(div_dim)
        data = math.sum(components, 0)
        return CenteredGrid(data, field.box, field.extrapolation.spatial_gradient())
    elif isinstance(field, CenteredGrid):
        left, right = shift(field, (-1, 1), stack_dim='div_')
        grad = (right - left) / (field.dx * 2)
        components = [grad.vector[i].div_[i] for i in range(grad.div_.size)]
        result = sum(components)
        return result
    else:
        raise NotImplementedError(f"{type(field)} not supported. Only StaggeredGrid allowed.")


FieldType = TypeVar('FieldType', bound=Field)
GridType = TypeVar('GridType', bound=Grid)


def minimize(function, x0: Grid, solve_params: math.Solve):
    data_function = _operate_on_values(function, x0)
    converged, x, iterations = math.minimize(data_function, x0.values, solve_params=solve_params)
    return converged, x0.with_(values=x), iterations


def solve(function, y: Grid, x0: Grid, solve_params: math.Solve, constants: tuple or list = (), callback=None):
    if callback is not None:
        def field_callback(x):
            x = x0.with_(values=x)
            callback(x)
    else:
        field_callback = None
    data_function = _operate_on_values(function, x0)
    constants = [c.values if isinstance(c, SampledField) else c for c in constants]
    assert all(isinstance(c, math.Tensor) for c in constants)
    converged, x, iterations = math.solve(data_function, y.values, x0.values, solve_params=solve_params, constants=constants, callback=field_callback)
    return converged, x0.with_(values=x), iterations


def _operate_on_values(field_function, *proto_fields):
    """
    Constructs a wrapper function operating on field values from a function operating on fields.
    The wrapper function assembles fields and calls `field_function`.

    This is useful when passing functions to a `phi.math` operation, e.g. `phi.math.solve()`.

    Args:
        field_function: Function whose arguments are fields
        *proto_fields: To specify non-value properties of the fields.

    Returns:
        Wrapper for `field_function` that takes the field values of as input and returns the field values of the result.
    """
    @wraps(field_function)
    def wrapper(*field_data):
        fields = [proto.with_(values=data) for data, proto in zip(field_data, proto_fields)]
        result = field_function(*fields)
        if isinstance(result, math.Tensor):
            return result
        elif isinstance(result, SampledField):
            return result.values
        else:
            raise ValueError(f"function must return an instance of SampledField or Tensor but returned {result}")
    return wrapper


def jit_compile(f: Callable):
    """
    Wrapper for `phi.math.jit_compile()` where `f` is a function operating on fields instead of tensors.

    Here, the arguments and output of `f` should be instances of `Field`.
    """
    INPUT_FIELDS = []
    OUTPUT_FIELDS = []

    def tensor_function(*tensors):
        fields = [field.with_(values=t) for field, t in zip(INPUT_FIELDS, tensors)]
        result = f(*fields)
        results = [result] if not isinstance(result, (tuple, list)) else result
        OUTPUT_FIELDS.clear()
        OUTPUT_FIELDS.extend(results)
        result_tensors = [field.values for field in results]
        return result_tensors

    tensor_trace = math.jit_compile(tensor_function)

    def wrapper(*fields):
        INPUT_FIELDS.clear()
        INPUT_FIELDS.extend(fields)
        tensors = [field.values for field in fields]
        result_tensors = tensor_trace(*tensors)
        result_tensors = [result_tensors] if not isinstance(result_tensors, (tuple, list)) else result_tensors
        result = [field.with_(values=t) for field, t in zip(OUTPUT_FIELDS, result_tensors)]
        return result[0] if len(result) == 1 else result

    return wrapper


def functional_gradient(f: Callable, wrt: tuple or list = (0,), get_output=False) -> Callable:
    """
    Wrapper for `phi.math.functional_gradient()` where `f` is a function operating on fields instead of tensors.

    Here, the arguments of `f` should be instances of `Field`.
    `f` returns a scalar tensor and optionally auxiliary fields.
    """
    INPUT_FIELDS = []
    OUTPUT_FIELDS = []

    def tensor_function(*tensors):
        fields = [field.with_(values=t) for field, t in zip(INPUT_FIELDS, tensors)]
        result = f(*fields)
        results = [result] if not isinstance(result, (tuple, list)) else result
        assert isinstance(results[0], math.Tensor)
        OUTPUT_FIELDS.clear()
        OUTPUT_FIELDS.extend(results)
        result_tensors = [r.values if isinstance(r, Field) else r for r in results]
        return result_tensors

    tensor_gradient = math.functional_gradient(tensor_function, wrt=wrt, get_output=get_output)

    def wrapper(*fields):
        INPUT_FIELDS.clear()
        INPUT_FIELDS.extend(fields)
        tensors = [field.values for field in fields]
        result_tensors = tuple(tensor_gradient(*tensors))
        proto_fields = []
        if get_output:
            proto_fields.extend(OUTPUT_FIELDS)
        proto_fields.extend([t for i, t in enumerate(INPUT_FIELDS) if i in wrt])
        result = [field.with_(values=t) if isinstance(field, Field) else t for field, t in zip(proto_fields, result_tensors)]
        return result

    return wrapper


def data_bounds(field: SampledField):
    data = field.points
    min_vec = math.min(data, dim=data.shape.spatial.names)
    max_vec = math.max(data, dim=data.shape.spatial.names)
    return Box(min_vec, max_vec)


def mean(field: Grid):
    return math.mean(field.values, field.shape.spatial)


def normalize(field: SampledField, norm: SampledField, epsilon=1e-5):
    data = math.normalize_to(field.values, norm.values, epsilon)
    return field.with_(values=data)


def pad(grid: Grid, widths: int or tuple or list or dict):
    if isinstance(widths, int):
        widths = {axis: (widths, widths) for axis in grid.shape.spatial.names}
    elif isinstance(widths, (tuple, list)):
        widths = {axis: (width if isinstance(width, (tuple, list)) else (width, width)) for axis, width in zip(grid.shape.spatial.names, widths)}
    else:
        assert isinstance(widths, dict)
    widths_list = [widths[axis] for axis in grid.shape.spatial.names]
    if isinstance(grid, Grid):
        data = math.pad(grid.values, widths, grid.extrapolation)
        w_lower = math.wrap([w[0] for w in widths_list])
        w_upper = math.wrap([w[1] for w in widths_list])
        box = Box(grid.box.lower - w_lower * grid.dx, grid.box.upper + w_upper * grid.dx)
        return type(grid)(data, box, grid.extrapolation)
    raise NotImplementedError(f"{type(grid)} not supported. Only Grid instances allowed.")


def downsample2x(grid: Grid) -> GridType:
    if isinstance(grid, CenteredGrid):
        values = math.downsample2x(grid.values, grid.extrapolation)
        return CenteredGrid(values, grid.bounds, grid.extrapolation)
    elif isinstance(grid, StaggeredGrid):
        values = []
        for dim, centered_grid in zip(grid.shape.spatial.names, grid.unstack()):
            odd_discarded = centered_grid.values[{dim: slice(None, None, 2)}]
            others_interpolated = math.downsample2x(odd_discarded, grid.extrapolation, dims=grid.shape.spatial.without(dim))
            values.append(others_interpolated)
        return StaggeredGrid(math.channel_stack(values, 'vector'), grid.bounds, grid.extrapolation)
    else:
        raise ValueError(type(grid))


def upsample2x(grid: GridType) -> GridType:
    if isinstance(grid, CenteredGrid):
        values = math.upsample2x(grid.values, grid.extrapolation)
        return CenteredGrid(values, grid.bounds, grid.extrapolation)
    elif isinstance(grid, StaggeredGrid):
        raise NotImplementedError()
    else:
        raise ValueError(type(grid))


def concat(*fields: SampledField, dim: str):
    assert all(isinstance(f, SampledField) for f in fields)
    assert all(isinstance(f, type(fields[0])) for f in fields)
    if any(f.extrapolation != fields[0].extrapolation for f in fields):
        raise NotImplementedError("Concatenating extrapolations not supported")
    if isinstance(fields[0], Grid):
        values = math.concat([f.values for f in fields], dim=dim)
        return fields[0].with_(values=values)
    elif isinstance(fields[0], PointCloud):
        elements = geom.concat([f.elements for f in fields], dim, sizes=[f.shape.get_size(dim) for f in fields])
        values = math.concat([math.expand(f.values, dim, f.shape.get_size(dim)) for f in fields], dim)
        colors = math.concat([math.expand(f.color, dim, f.shape.get_size(dim)) for f in fields], dim)
        return fields[0].with_(elements=elements, values=values, color=colors)
    raise NotImplementedError(type(fields[0]))


def batch_stack(*fields, dim: str):
    assert all(isinstance(f, SampledField) for f in fields)
    assert all(isinstance(f, type(fields[0])) for f in fields)
    if any(f.extrapolation != fields[0].extrapolation for f in fields):
        raise NotImplementedError("Concatenating extrapolations not supported")
    if isinstance(fields[0], Grid):
        values = math.batch_stack([f.values for f in fields], dim)
        return fields[0].with_(values=values)
    elif isinstance(fields[0], PointCloud):
        elements = geom.stack(*[f.elements for f in fields], dim=dim)
        values = math.batch_stack([f.values for f in fields], dim=dim)
        colors = math.batch_stack([f.color for f in fields], dim=dim)
        return fields[0].with_(elements=elements, values=values, color=colors)
    raise NotImplementedError(type(fields[0]))


def abs(x: SampledField) -> SampledField:
    return x._op1(math.abs)


def sign(x: SampledField) -> SampledField:
    return x._op1(math.sign)


def round_(x: SampledField) -> SampledField:
    return x._op1(math.round)


def ceil(x: SampledField) -> SampledField:
    return x._op1(math.ceil)


def floor(x: SampledField) -> SampledField:
    return x._op1(math.floor)


def sqrt(x: SampledField) -> SampledField:
    return x._op1(math.sqrt)


def exp(x: SampledField) -> SampledField:
    return x._op1(math.exp)


def isfinite(x: SampledField) -> SampledField:
    return x._op1(math.isfinite)


def real(field: SampledField):
    return field._op1(math.real)


def imag(field: SampledField):
    return field._op1(math.imag)


def sin(x: SampledField) -> SampledField:
    return x._op1(math.sin)


def cos(x: SampledField) -> SampledField:
    return x._op1(math.cos)


def cast(x: SampledField, dtype: DType) -> SampledField:
    return x._op1(partial(math.cast, dtype=dtype))


def assert_close(*fields: SampledField or math.Tensor or Number,
                 rel_tolerance: float = 1e-5,
                 abs_tolerance: float = 0):
    """ Raises an AssertionError if the `values` of the given fields are not close. See `phi.math.assert_close()`. """
    f0 = next(filter(lambda t: isinstance(t, SampledField), fields))
    values = [(f >> f0).values if isinstance(f, SampledField) else math.wrap(f) for f in fields]
    math.assert_close(*values, rel_tolerance=rel_tolerance, abs_tolerance=abs_tolerance)


# def staggered_curl_2d(grid, pad_width=(1, 2)):
#     assert isinstance(grid, CenteredGrid)
#     kernel = math.zeros((3, 3, 1, 2))
#     kernel[1, :, 0, 0] = [0, 1, -1]  # y-component: - dz/dx
#     kernel[:, 1, 0, 1] = [0, -1, 1]  # x-component: dz/dy
#     scalar_potential = grid.padded([pad_width, pad_width]).values
#     vector_field = math.conv(scalar_potential, kernel, padding='valid')
#     return StaggeredGrid(vector_field, bounds=grid.box)


def where(mask: Field or Geometry, field_true: Field, field_false: Field):
    if isinstance(mask, Geometry):
        mask = HardGeometryMask(mask)
    elif isinstance(mask, SampledField):
        field_true = field_true.at(mask)
        field_false = field_false.at(mask)
    elif isinstance(field_true, SampledField):
        mask = mask.at(field_true)
        field_false = field_false.at(field_true)
    elif isinstance(field_false, SampledField):
        mask = mask.at(field_true)
        field_true = field_true.at(mask)
    else:
        raise NotImplementedError('At least one argument must be a SampledField')
    values = mask.values * field_true.values + (1 - mask.values) * field_false.values
    # values = math.where(mask.values, field_true.values, field_false.values)
    return field_true.with_(values=values)


def l2_loss(field: SampledField, batch_norm=True):
    """ L2 loss for the unweighted values of the field. See `phi.math.l2_loss()`. """
    return math.l2_loss(field.values, batch_norm=batch_norm)


def stop_gradient(field: SampledField):
    """ See `phi.math.stop_gradient()` """
    return field._op1(math.stop_gradient)


def extrapolate_valid(grid: GridType, valid: GridType, distance_cells=1) -> tuple:
    """
    Extrapolates values of `grid` which are marked by nonzero values in `valid` using `phi.math.extrapolate_valid_values().
    If `values` is a StaggeredGrid, its components get extrapolated independently.

    Args:
        grid: Grid holding the values for extrapolation
        valid: Grid (same type as `values`) marking the positions for extrapolation with nonzero values
        distance_cells: Number of extrapolation steps

    Returns:
        grid: Grid with extrapolated values.
        valid: binary Grid marking all valid values after extrapolation.
    """
    assert isinstance(valid, type(grid)), 'Type of valid Grid must match type of grid.'
    if isinstance(grid, CenteredGrid):
        new_values, new_valid = extrapolate_valid_values(grid.values, valid.values, distance_cells)
        return grid.with_(values=new_values), valid.with_(values=new_valid)
    elif isinstance(grid, StaggeredGrid):
        new_values = []
        new_valid = []
        for cgrid, cvalid in zip(grid.unstack('vector'), valid.unstack('vector')):
            new_tensor, new_mask = extrapolate_valid(cgrid, valid=cvalid, distance_cells=distance_cells)
            new_values.append(new_tensor.values)
            new_valid.append(new_mask.values)
        return grid.with_(values=math.channel_stack(new_values, 'vector')), valid.with_(values=math.channel_stack(new_valid, 'vector'))
    else:
        raise NotImplementedError()
