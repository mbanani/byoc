import torch
from pytorch3d import transforms as pt3d_T


def random_Rt(batch_size: float, max_R: float, max_t: float, device):
    # Generate random numbers from 0 to max_R
    r_mag = torch.rand(batch_size, 1) * max_R
    q_w = r_mag.cos()
    q_xyz = torch.randn(batch_size, 3)
    q_xyz = (q_xyz / q_xyz.norm(p=2, dim=1, keepdim=True)) * r_mag.sin()
    q = torch.cat((q_w, q_xyz), dim=1)
    R = pt3d_T.quaternion_to_matrix(q)

    # get t; scale it
    t = torch.randn(batch_size, 3, 1)
    t = t * max_t / t.norm(p=2, dim=1, keepdim=True)
    t = t * torch.rand(batch_size, 1, 1)

    # concat
    Rt = torch.cat((R, t), dim=2).to(device)

    return Rt


# @torch.jit.script
def transform_points_Rt(
    points: torch.Tensor, viewpoint: torch.Tensor, inverse: bool = False
):
    N, H, W = viewpoint.shape
    assert H == 3 and W == 4, "Rt is B x 3 x 4 "
    t = viewpoint[:, :, 3]
    r = viewpoint[:, :, 0:3]

    # transpose r to handle the fact that P in num_points x 3
    # yT = (RX) + t = (XT @ RT) + t
    r = r.transpose(1, 2).contiguous()

    # invert if needed
    if inverse:
        points = points - t[:, None, :]
        points = points.bmm(r.inverse())
    else:
        points = points.bmm(r)
        points = points + t[:, None, :]

    return points
