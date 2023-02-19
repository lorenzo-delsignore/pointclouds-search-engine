import torch

class ChanferLoss3d(torch.nn.modules.Module):
    """
    GPU chanferloss
    """

    def __init__(self):
        super().__init__()
        from ChamferDistancePytorch.chamfer3D import dist_chamfer_3D
        self.chamfer_raw = dist_chamfer_3D.chamfer_3DDist()

    def forward(self, pc_a, pc_b):
        """ Compute the chamfer loss for batched pointclouds.
        :param pc_a: torch.Tensor B x Na-points per point-cloud x 3
        :param pc_b: torch.Tensor B x Nb-points per point-cloud x 3
        :return: B floats, indicating the chamfer distances
        """
        dist_a, dist_b, idx_a, idx_b = self.chamfer_raw(pc_a, pc_b)
        dist = dist_a.mean(1) + dist_b.mean(1) # reduce separately, sizes of points can be different
        return dist.sum()

class ChanferLoss(torch.nn.modules.Module):
    """
    CPU chanferloss
    """

    def huber_loss(self, error, delta=1.0):
        import torch

        """
        Args:
            error: Torch tensor (d1,d2,...,dk)
        Returns:
            loss: Torch tensor (d1,d2,...,dk)
        x = error = pred - gt or dist(pred,gt)
        0.5 * |x|^2                 if |x|<=d
        0.5 * d^2 + d * (|x|-d)     if |x|>d
        Ref: https://github.com/charlesq34/frustum-pointnets/blob/master/models/model_util.py
        """
        abs_error = torch.abs(error)
        quadratic = torch.clamp(abs_error, max=delta)
        linear = (abs_error - quadratic)
        loss = 0.5 * quadratic**2 + delta * linear
        return loss

    def nn_distance(self, pc1, pc2, l1smooth=False, delta=1.0, l1=False):
        import torch

        """
        Input:
            pc1: (B,N,C) torch tensor
            pc2: (B,M,C) torch tensor
            l1smooth: bool, whether to use l1smooth loss
            delta: scalar, the delta used in l1smooth loss
        Output:
            dist1: (B,N) torch float32 tensor
            idx1: (B,N) torch int64 tensor
            dist2: (B,M) torch float32 tensor
            idx2: (B,M) torch int64 tensor
        """
        N = pc1.shape[1]
        M = pc2.shape[1]
        pc1_expand_tile = pc1.unsqueeze(2).repeat(1,1,M,1)
        pc2_expand_tile = pc2.unsqueeze(1).repeat(1,N,1,1)
        pc_diff = pc1_expand_tile - pc2_expand_tile

        if l1smooth:
            pc_dist = torch.sum(self.huber_loss(pc_diff, delta), dim=-1) # (B,N,M)
        elif l1:
            pc_dist = torch.sum(torch.abs(pc_diff), dim=-1) # (B,N,M)
        else:
            pc_dist = torch.sum(pc_diff**2, dim=-1) # (B,N,M)

        dist1, idx1 = torch.min(pc_dist, dim=2) # (B,N)
        dist2, idx2 = torch.min(pc_dist, dim=1) # (B,M)
        return dist1, idx1, dist2, idx2

    def forward(self, pc_a, pc_b):
        pc_b = pc_b.view(-1, 2048, 3)
        dist_a, _, dist_b, _ = self.nn_distance(pc_a, pc_b)
        dist = dist_a.mean(1) + dist_b.mean(1) # reduce separately, sizes of points can be different
        return dist.sum()

class PointNetLoss(torch.nn.modules.Module):
    """
    PointNetLoss
    """

    def loss(self, outputs, labels, m3x3, m64x64, alpha = 0.0001):
        criterion = torch.nn.NLLLoss()
        bs=outputs.size(0)
        id3x3 = torch.eye(3, requires_grad=True).repeat(bs,1,1)
        id64x64 = torch.eye(64, requires_grad=True).repeat(bs,1,1)
        if outputs.is_cuda:
            id3x3=id3x3.cuda()
            id64x64=id64x64.cuda()
        diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
        diff64x64 = id64x64-torch.bmm(m64x64,m64x64.transpose(1,2))
        crit_loss = criterion(outputs, labels)
        return crit_loss + alpha * (torch.norm(diff3x3)+torch.norm(diff64x64)) / float(bs)

    def forward(self, y_pred, y_train):
        outputs, m3x3, m64x64 = y_pred
        return self.loss(outputs, y_train, m3x3, m64x64)
