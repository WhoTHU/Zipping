import numpy as np
import torch
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm
from scipy.sparse import csr_matrix

def compute_matrixes(mesh):
    verts = mesh.verts_list()[0].detach()
    faces = mesh.faces_list()[0].detach()
    verts_uvs = mesh.textures.verts_uvs_list()[0].detach()
    faces_uvs = mesh.textures.faces_uvs_list()[0].detach()
    num_verts = len(verts)

    # check adjacent points
    adjacent = torch.Tensor(size=[num_verts, num_verts]).to(mesh.device).fill_(0)
    for f in faces:
        adjacent[f[0], f[1]], adjacent[f[1], f[2]], adjacent[f[2], f[0]] = 1, 1, 1
    adjacent = adjacent + adjacent.t()
    row_ind, col_ind = [x.squeeze(1) for x in adjacent.nonzero().split(1, dim=1)]
    data = (verts_uvs[row_ind] - verts_uvs[col_ind]).norm(2, 1)

    # check overlap points
    dist = (verts.unsqueeze(0) - verts.unsqueeze(1)).abs().sum(-1) + torch.diag(verts.new_ones(num_verts))
    ri, ci = [x.squeeze(1) for x in (dist == 0).nonzero().split(1, dim=1)]
    row_ind = torch.cat([row_ind, ri])
    col_ind = torch.cat([col_ind, ci])
    data = torch.cat([data, torch.zeros_like(ri)])
    overlap_set_all = (dist == 0).nonzero()
    overlap_set_all = overlap_set_all[overlap_set_all[:, 0] < overlap_set_all[:, 1]]
    # adjacent[ri, ci] = 1

    adjacent = adjacent.clamp(0, 1).bool()
    # edges = torch.stack([row_ind, col_ind, data], 1)

    row_ind, col_ind, data = row_ind.detach().cpu().numpy(), col_ind.detach().cpu().numpy(), data.detach().cpu().numpy()
    distance_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(num_verts, num_verts))
    return adjacent, distance_matrix

def show_graph(adj, locations_set, overlap_set=None, inds_groups=None, figsize=(10, 10), s=26.0, axis_equal=True, axis_off=True, colors=None):
    color_set = plt.cm.tab10
    grap_num = len(locations_set)

    if colors is None:
        if inds_groups is not None:
            inds_all = np.concatenate(inds_groups)
            colors = np.zeros([inds_all.max() + 1, 4])
            for i, inds in enumerate(inds_groups):
                colors[inds] = color_set(i)
            colors = colors[inds_all]
        else:
            colors = np.array(color_set(0)).reshape(1, 4)

    if adj is not None:
        edges = adj.nonzero()
        edges = edges[edges[:, 0] < edges[:, 1]]

    fig = plt.figure(figsize=figsize)
    # fig, axs = plt.subplots(1, grap_num, figsize=figsize)
    for i, locations in enumerate(locations_set):
        ax = fig.add_subplot(1, len(locations_set), i+1)
        #     s = (locations[:, 0].max() - locations[:, 0].min()) * (locations[:, 1].max() - locations[:, 1].min())* 1e0 / len(locations)
        #     s = s.item()
        if adj is not None:
            points = torch.stack([locations[edges[:, 0]], locations[edges[:, 1]]], -1)
            xs = points[:, 0, :].cpu().numpy().T
            ys = points[:, 1, :].cpu().numpy().T
            ax.plot(xs, ys, c=(0.9, 0.6, 0.0), linewidth=s / 20, alpha=0.9)
        ax.scatter(locations[:, 0].cpu().numpy(), locations[:, 1].cpu().numpy(), s=s, c=colors)

        if overlap_set is not None:
            inds = overlap_set[:, 0]
            inds = inds[inds < len(locations)]
            if len(inds) < len(overlap_set):
                print('Omit %d red pints' % (overlap_set[:, 0] - len(inds)))
            ax.scatter(locations[inds, 0].cpu().numpy(), locations[inds, 1].cpu().numpy(), s=s, c='r')
            inds = overlap_set[:, 1]
            inds = inds[inds < len(locations)]
            if len(inds) < len(overlap_set):
                print('Omit %d black pints' % (overlap_set[:, 0] - len(inds)))
            ax.scatter(locations[inds, 0].cpu().numpy(), locations[inds, 1].cpu().numpy(), s=s, c='k')
        if axis_equal:
            ax.set_aspect('equal')
        if axis_off:
            ax.set_axis_off()
    return fig, ax


def get_force(locations, units, overlap_set, perp_norm_ori, v3_mat, d_ori, alpha, beta, gamma, step_size_max, overlap_md, accumulate_overlap):
    force = torch.zeros_like(locations)

    #     1.1 compute perpendicular force from each triangle
    locs = locations[units.view(-1)].view(-1, 3, 2)
    vertexes = torch.cat([locs, locs], 1)
    rel_vec = vertexes[:, 1:] - vertexes[:, :-1]
    #     perp vector is (- v2v3*v1 + v1v3*v2)/(v3v3)
    perp = ((rel_vec[:, 1:4] * rel_vec[:, 0:3]).sum(-1, keepdim=True) * rel_vec[:, 2:5] - (
                rel_vec[:, 1:4] * rel_vec[:, 2:5]).sum(-1, keepdim=True) * rel_vec[:, 0:3]) / (
                       rel_vec[:, 1:4] * rel_vec[:, 1:4]).sum(-1, keepdim=True)
    perp_norm = perp.norm(2, -1, keepdim=True)
    force_together = - alpha[0] * (perp_norm_ori / perp_norm - 1) / perp_norm * perp
    #     force_together = alpha[0] * (perp_norm / perp_norm_ori).log() / perp_norm * perp

    #     1.2 shear force
    i3 = v3_mat.matmul(rel_vec[:, 1:4].unsqueeze(-1)).squeeze(-1) / rel_vec[:, 1:4].norm(2, -1, keepdim=True)
    i3t = i3.flip(-1) * i3.new([-1, 1])
    R = torch.stack([i3, i3t], -1)
    d = R.matmul(d_ori.unsqueeze(-1)).squeeze(-1) * perp_norm / perp_norm_ori
    fine_loc = (vertexes[:, 1:4] + vertexes[:, 2:5]) * 0.5 + d
    force_together += alpha[1] * (fine_loc - vertexes[:, 0:3]) / perp_norm

    #     1.-2 compute for every vertex
    force_together -= force_together.mean(1, keepdim=True)
    force.index_put_((units.view(-1),), force_together.view(-1, 2), accumulate=True)

    #     1.-1 force for overlapped points
    locs_o = locations[overlap_set.view(-1)].view(-1, 2, 2)
    over_force = beta * (locs_o.flip(1) - locs_o)
    if overlap_md > 0:
        suc_inds = (over_force[:, 0].norm(2, -1) / beta) < overlap_md
        over_force[suc_inds] = force[overlap_set[suc_inds].view(-1)].view(-1, 2, 2).sum(1, keepdim=True)
    #     compute for every vertex
    #     force.index_put_((overlap_set.view(-1), ), over_force.view(-1, 2), accumulate=True)
    force.index_put_((overlap_set.view(-1),), over_force.view(-1, 2), accumulate=accumulate_overlap)

    #     2.1 decide step size
    force_dir = force[units.view(-1)].view(-1, 3, 2)
    force_dir = torch.cat([force_dir, force_dir], 1)
    fdiff = force_dir[:, 1:] - force_dir[:, :-1]
    #     rotate the vector, therefore cross product becomes dot product
    fdiff = torch.stack([fdiff[:, 0:3].flip(-1) * fdiff.new([1, -1]), fdiff[:, 2:5]], 2).view(-1, 2, 2)
    vdiff = torch.stack([rel_vec[:, 0:3].flip(-1) * rel_vec.new([1, -1]), rel_vec[:, 2:5]], 2).view(-1, 2, 2)
    #     solve (v1'+f1'*t)(v2+f2*t)=0, cross product
    a = (fdiff[:, 0] * fdiff[:, 1]).sum(-1)
    b = (fdiff[:, 0] * vdiff[:, 1]).sum(-1) + (fdiff[:, 1] * vdiff[:, 0]).sum(-1)
    c = (vdiff[:, 0] * vdiff[:, 1]).sum(-1)
    delta = b * b - 4 * a * c
    solving = torch.stack([a, b, c, delta], 1)
    step_size = step_size_max
    s1 = solving[(solving[:, 0] == 0).logical_and(solving[:, 1] != 0)]  # a=0, b!=0
    if len(s1) > 0:
        t = -s1[:, 2] / s1[:, 1]
        t = t[t > 0]
        if len(t) > 0:
            step_size = min(step_size, t.min().item() * gamma)
    s2 = solving[(solving[:, 0] != 0).logical_and(solving[:, 3] >= 0)]  # a!=0, delta>0
    delta_sqrt = s2[:, 3].sqrt()
    t = torch.cat([(-s2[:, 1] + delta_sqrt) / (2 * s2[:, 0]), (-s2[:, 1] - delta_sqrt) / (2 * s2[:, 0])])
    t = t[t > 0]
    if len(t) > 0:
        step_size = min(step_size, t.min().item() * gamma)
    #         print(t)
    #         raise ValueError
    return force, step_size


def move_by_force(locations_ori, locations_init, units, overlap_set, alpha=(1.0,) * 3, beta=1.0, gamma=0.5, step_size_max=0.1, steps=1000, overlap_md=1e-4, accumulate_overlap=True, retrieve_history=False):
    locations_history = [locations_init.detach().cpu().clone()] if retrieve_history else None
    # compute ori perpendicular line
    locations = locations_ori
    locs = locations[units.view(-1)].view(-1, 3, 2)
    vertexes = torch.cat([locs, locs], 1)
    rel_vec = vertexes[:, 1:] - vertexes[:, :-1]
    perp_ori = ((rel_vec[:, 1:4] * rel_vec[:, 0:3]).sum(-1, keepdim=True) * rel_vec[:, 2:5] - (rel_vec[:, 1:4] * rel_vec[:, 2:5]).sum(-1, keepdim=True) * rel_vec[:, 0:3]) / (rel_vec[:, 1:4] * rel_vec[:, 1:4]).sum(-1, keepdim=True)
    perp_norm_ori = perp_ori.norm(2, -1, keepdim=True)

    # compute ori edge length
    edge_norm_ori = rel_vec.norm(2, -1, keepdim=True)

    # compute ori relative position d
    d_ori = (rel_vec[:, 2:5] - rel_vec[:, 0:3]) * 0.5
    v3_mat = torch.stack([rel_vec[:, 1:4], rel_vec[:, 1:4].flip(-1) * rel_vec.new([-1, 1])], -2) / edge_norm_ori[:, 1:4].unsqueeze(-1)

    F = partial(get_force, units=units, overlap_set=overlap_set, perp_norm_ori=perp_norm_ori, v3_mat=v3_mat, d_ori=d_ori, alpha=alpha, beta=beta, gamma=gamma, step_size_max=step_size_max, overlap_md=overlap_md, accumulate_overlap=accumulate_overlap)

    step_log = []
    locations = locations_init
    # iterations
    for i in tqdm(range(steps)):
        force, step_size = F(locations)
    #     force, step_size = F(1, 0)
        step_log.append(step_size)
        locations = locations + force * step_size
        if retrieve_history:
            locations_history.append(locations.detach().cpu().clone())
    info = {
        'step_log': step_log,
        'locations_history' : locations_history,
        }

    return locations, info

def grouping_verts(mesh):
    verts_uvs = mesh.textures.verts_uvs_list()[0]
    num_verts = len(verts_uvs)
    adjacent = torch.Tensor(size=[num_verts, num_verts]).to(mesh.device).fill_(0)
    for f in mesh.faces_list()[0]:
        adjacent[f[0], f[1]], adjacent[f[1], f[2]], adjacent[f[2], f[0]] = 1, 1, 1

    adjacent = adjacent + adjacent.t()
    k = len(adjacent.nonzero())
    n = 0
    while True:
        n += 1
        adjacent = adjacent.matmul(adjacent).clamp_(max=1)
        k_new = len(adjacent.nonzero())
        if k == k_new:
            print('Ends in %d iters' % n)
            break
        else:
            k = k_new
    adjacent += torch.diag(torch.ones(num_verts)).to(mesh.device)
    groups = []
    indicator = torch.ones(num_verts)
    while indicator.sum() > 0:
        first_ind = indicator.nonzero()[0][0]
        indexs = adjacent[first_ind] > 0
        groups.append(indexs.nonzero()[:, 0])
        indicator[indexs] = 0
    print('Devided in %d groups' % (len(groups)))
    return groups