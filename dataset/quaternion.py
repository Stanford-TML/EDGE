import torch
from pytorch3d.transforms import (axis_angle_to_matrix, matrix_to_axis_angle,
                                  matrix_to_quaternion, matrix_to_rotation_6d,
                                  quaternion_to_matrix, rotation_6d_to_matrix)


def quat_to_6v(q):
    assert q.shape[-1] == 4
    mat = quaternion_to_matrix(q)
    mat = matrix_to_rotation_6d(mat)
    return mat


def quat_from_6v(q):
    assert q.shape[-1] == 6
    mat = rotation_6d_to_matrix(q)
    quat = matrix_to_quaternion(mat)
    return quat


def ax_to_6v(q):
    assert q.shape[-1] == 3
    mat = axis_angle_to_matrix(q)
    mat = matrix_to_rotation_6d(mat)
    return mat


def ax_from_6v(q):
    assert q.shape[-1] == 6
    mat = rotation_6d_to_matrix(q)
    ax = matrix_to_axis_angle(mat)
    return ax


def quat_slerp(x, y, a):
    """
    Performs spherical linear interpolation (SLERP) between x and y, with proportion a

    :param x: quaternion tensor (N, S, J, 4)
    :param y: quaternion tensor (N, S, J, 4)
    :param a: interpolation weight (S, )
    :return: tensor of interpolation results
    """
    len = torch.sum(x * y, axis=-1)

    neg = len < 0.0
    len[neg] = -len[neg]
    y[neg] = -y[neg]

    a = torch.zeros_like(x[..., 0]) + a

    amount0 = torch.zeros_like(a)
    amount1 = torch.zeros_like(a)

    linear = (1.0 - len) < 0.01
    omegas = torch.arccos(len[~linear])
    sinoms = torch.sin(omegas)

    amount0[linear] = 1.0 - a[linear]
    amount0[~linear] = torch.sin((1.0 - a[~linear]) * omegas) / sinoms

    amount1[linear] = a[linear]
    amount1[~linear] = torch.sin(a[~linear] * omegas) / sinoms

    # reshape
    amount0 = amount0[..., None]
    amount1 = amount1[..., None]

    res = amount0 * x + amount1 * y

    return res
