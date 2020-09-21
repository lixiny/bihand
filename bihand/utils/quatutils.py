import torch
import torch.nn.functional as torch_f


def normalize_quaternion(quaternion: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    r"""Normalizes a quaternion.
    The quaternion should be in (w, x, y, z) format.

    Args:
        quaternion (torch.Tensor): a tensor containing a quaternion to be
          normalized. The tensor can be of shape :math:`(*, 4)`.
        eps (Optional[bool]): small value to avoid division by zero.
          Default: 1e-12.

    Return:
        torch.Tensor: the normalized quaternion of shape :math:`(*, 4)`.

    Example:
        >>> quaternion = torch.tensor([1., 0., 1., 0.])
        >>> kornia.normalize_quaternion(quaternion)
        tensor([0.7071, 0.0000, 0.7071, 0.0000])
    """
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape (*, 4). Got {}".format(quaternion.shape))
    return torch_f.normalize(quaternion, p=2, dim=-1, eps=eps)


def quaternion_inv(q):
    """
    inverse quaternion(s) q
    The quaternion should be in (w, x, y, z) format.
    Expects  tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4

    q_conj = q[..., 1:] * -1.0
    q_conj = torch.cat((q[..., 0:1], q_conj), dim=-1)
    q_norm = torch.norm(q, dim=-1, keepdim=True)
    return q_conj / q_norm


def quaternion_mul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    The quaternion should be in (w, x, y, z) format.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    # terms; ( * , 4, 4)
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] + terms[:, 2, 3] - terms[:, 3, 2]
    y = terms[:, 0, 2] - terms[:, 1, 3] + terms[:, 2, 0] + terms[:, 3, 1]
    z = terms[:, 0, 3] + terms[:, 1, 2] - terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


# source code from https://kornia.readthedocs.io/en/latest/_modules/kornia/geometry/conversions.html
# Fix bugs in Kornia ( xyzw -> wxyz)
def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert quaternion vector to angle axis of rotation.
    The quaternion should be in (w, x, y, z) format.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = kornia.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}".format(quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]  # x
    q2: torch.Tensor = quaternion[..., 2]  # y
    q3: torch.Tensor = quaternion[..., 3]  # z
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]  # w
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0, torch.atan2(-sin_theta, -cos_theta), torch.atan2(sin_theta, cos_theta)
    )

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


# source code from https://kornia.readthedocs.io/en/latest/_modules/kornia/geometry/conversions.html
# Fix bugs in Kornia ( xyzw -> wxyz)
def angle_axis_to_quaternion(angle_axis: torch.Tensor) -> torch.Tensor:
    r"""Convert an angle axis to a quaternion.
    The quaternion vector has components in (w, x, y, z) format.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        angle_axis (torch.Tensor): tensor with angle axis.

    Return:
        torch.Tensor: tensor with quaternion.

    Shape:
        - Input: :math:`(*, 3)` where `*` means, any number of dimensions
        - Output: :math:`(*, 4)`

    Example:
        >>> angle_axis = torch.rand(2, 4)  # Nx4
        >>> quaternion = kornia.angle_axis_to_quaternion(angle_axis)  # Nx3
    """
    if not torch.is_tensor(angle_axis):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(angle_axis)))

    if not angle_axis.shape[-1] == 3:
        raise ValueError("Input must be a tensor of shape Nx3 or 3. Got {}".format(angle_axis.shape))
    # unpack input and compute conversion
    a0: torch.Tensor = angle_axis[..., 0:1]
    a1: torch.Tensor = angle_axis[..., 1:2]
    a2: torch.Tensor = angle_axis[..., 2:3]
    theta_squared: torch.Tensor = a0 * a0 + a1 * a1 + a2 * a2

    theta: torch.Tensor = torch.sqrt(theta_squared)
    half_theta: torch.Tensor = theta * 0.5

    mask: torch.Tensor = theta_squared > 0.0
    ones: torch.Tensor = torch.ones_like(half_theta)

    k_neg: torch.Tensor = 0.5 * ones
    k_pos: torch.Tensor = torch.sin(half_theta) / theta
    k: torch.Tensor = torch.where(mask, k_pos, k_neg)
    w: torch.Tensor = torch.where(mask, torch.cos(half_theta), ones)

    quaternion: torch.Tensor = torch.zeros_like(angle_axis)
    quaternion[..., 0:1] += a0 * k
    quaternion[..., 1:2] += a1 * k
    quaternion[..., 2:3] += a2 * k
    return torch.cat([w, quaternion], dim=-1)  # wxyz format
