import torch
import numpy as np
from easydict import EasyDict
from graphic_utils import *
import matplotlib.pyplot as plt

def prepare_mesh(mesh):
    adjacent, distance_matrix = compute_matrixes(mesh)
    verts = mesh.verts_list()[0].detach()
    faces = mesh.faces_list()[0].detach()
    verts_uvs = mesh.textures.verts_uvs_list()[0].detach()
    faces_uvs = mesh.textures.faces_uvs_list()[0].detach()

    # group the mesh verts
    groups_tshirt = grouping_verts(mesh)
    groups_tshirt_np = [g.cpu().numpy() for g in groups_tshirt]
    groups = groups_tshirt

    mesh_infos = EasyDict({
        'mesh': mesh,
        'verts': verts,
        'faces': faces,
        'verts_uvs': verts_uvs,
        'faces_uvs': faces_uvs,
        'adjacent': adjacent,
        'distance_matrix': distance_matrix,
        'groups': groups,
    })
    return mesh_infos


def prepare_part(args, mesh_infos, device, bin_size=None):
    inds = torch.cat([mesh_infos.groups[i] for i in args.part_ids])
    num = len(inds)
    print(f"Number of the points: {num}")

    verts, faces, verts_uvs, faces_uvs, distance_matrix, adjacent = mesh_infos.verts, mesh_infos.faces, mesh_infos.verts_uvs, mesh_infos.faces_uvs, mesh_infos.distance_matrix, mesh_infos.adjacent

    locations = verts_uvs.clone()
    dists_ori = torch.from_numpy(distance_matrix.todense()).to(device)
    adj = adjacent
    units = faces_uvs.clone()

    locations = locations[inds]
    dists_ori = dists_ori[inds][:, inds]
    adj = adj[inds][:, inds]
    units = units[(units[:, 0] >= inds.min()).logical_and(units[:, 0] <= inds.max())]
    assert (units.min() == inds.min()) and (units.max() == inds.max())
    units = units - inds.min()

    # compute overlap_set
    if bin_size is None: # search directly
        dist = (verts[inds].unsqueeze(0) - verts[inds].unsqueeze(1)).abs().sum(-1) + torch.diag(verts[inds].new_ones(len(inds)))
        overlap_set = (dist == 0).nonzero()
        overlap_set = overlap_set[overlap_set[:, 0] < overlap_set[:, 1]]
    else: # search in bins, in case the memory is too big
        bin_size = 1000
        verts_bin = verts.split(bin_size, 0)
        overlap_set = []
        for i in range(len(verts_bin)):
            for j in range(i, len(verts_bin)):
                dist = (verts_bin[i].unsqueeze(1) - verts_bin[j].unsqueeze(0)).abs().sum(-1)
                oset = (dist == 0).nonzero()
                oset[:, 0] = oset[:, 0] + i * bin_size
                oset[:, 1] = oset[:, 1] + j * bin_size
                oset = oset[oset[:, 0] < oset[:, 1]]
                overlap_set.append(oset)
        overlap_set = torch.cat(overlap_set, 0)
    return adj, overlap_set, locations, units

def pipeline(name, args, args_optim, mesh_infos, device, if_show=None, retrieve_history=False):
    adj, overlap_set, locations, units = prepare_part(args, mesh_infos, device, args.get('bin_size', None))

    # initialize the locations
    locations, locations_ori = args.init_fun(locations, device)

    if args.get('if_skip_move', False):
        print("Skip Zipping")
        info = None
    else:
        locations, info = move_by_force(locations_ori, locations, units, overlap_set, retrieve_history=retrieve_history, **args_optim)
        print(f"Actual time: {np.sum(info['step_log'])}")
        plt.plot(np.log(info['step_log']))
        plt.show()


    if if_show == 'single':
        fig = show_graph(adj, [locations], overlap_set, figsize=[17, 10], axis_equal=True, axis_off=False)
    elif if_show == 'double':
        fig = show_graph(None, [locations_ori, locations], overlap_set, figsize=[17, 10], axis_equal=True, axis_off=False)
    return locations, locations_ori, info, adj, overlap_set





