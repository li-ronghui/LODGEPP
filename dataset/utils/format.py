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