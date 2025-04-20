import os
import torch
import numpy as np
import pytorch3d.io as p3dio
from easydict import EasyDict
from functools import partial

from graphic_utils import *
from optim_utils import *

device = torch.device("cuda:0")
if not os.path.exists("./cache"):
    os.makedirs("./cache")
    
paths = ['./objs/Man_join/man.obj',
        './objs/tshirt_join/tshirt.obj',
        './objs/trouser_join/trouser.obj']

mesh_man, mesh_tshirt, mesh_trouser = [p3dio.load_objs_as_meshes([path], device=device) for path in paths]
print('Done')

def loc_init_front_back(locations, device):
    # initialize the locations for front&back
    locations[:, 0] = locations[:, 0] - locations[:, 0].mean()
    locations[:, 1] = locations[:, 1] - locations[:, 1].min()
    theta0 = np.pi / 2 * 0.95
    R = locations[:, 0].max() / theta0
    theta =  locations[:, 0] / R

    R = 10.0 - locations[:, 1]
    # R = R ** 0.5

    gamma = 1.3
    R0 = R.min() ** gamma
    R1 = R.max() ** gamma
    r0 = torch.tensor(0.5, device=device) ** gamma
    r1 = torch.tensor(4.0, device=device) ** gamma
    b = R0 - (R1 - R0) / (r1/r0 -1)
    a = r0 / (R0 - b)
    r = (a * (R**gamma - b)) ** (1 / gamma)
    locations = torch.stack([r * theta.sin(), r * theta.cos()], 1)
    locations_ori = locations.clone()
    return locations, locations_ori

def loc_init_sleeves(locations, device):
    # initialize the locations for sleeves
    locations[:, 0] = locations[:, 0] - locations[:, 0].mean()
    locations[:, 1] = locations[:, 1] - locations[:, 1].min()
    theta0 = np.pi * 0.95
    # locations[:, 0] *= (locations[:, 1] + 0.2)
    m = (locations[0] + locations[75]).abs()[0] / 2
    a = (locations[0] - locations[75]).abs()[0] / 2 / theta0
    b = ((locations[25] - locations[50]).abs()[0] / 2 / theta0 - a) / (locations[25] - locations[0]).abs()[1]
    R = a + locations[:, 1] * b
    theta =  (locations[:, 0] - m) / R
    # r = a * (R^2 - R0^2) ^ 1/2
    gamma = 0.7
    R0 = R.min() ** gamma
    R1 = R.max() ** gamma
    r0 = torch.tensor(0.5, device=device) ** gamma
    r1 = torch.tensor(4.0, device=device) ** gamma
    b = R0 - (R1 - R0) / (r1/r0 -1)
    a = r0 / (R0 - b)
    r = (a * (R**gamma - b)) ** (1 / gamma)
    locations = torch.stack([r * theta.sin(), r * theta.cos()], 1)
    locations_ori = locations.clone()
    return locations, locations_ori

def loc_init_fandb(locations, device):
    locations1 = torch.load('./cache/part_front.pt')
    locations2 = torch.load('./cache/part_back.pt')

    locations = torch.cat([locations1, -locations2], 0)
    locations_ori = locations.clone()
    return locations, locations_ori

def loc_init_tshirt_all(locations, device):
    # import and initialize
    locations1 = torch.load('./cache/part_fandb.pt')
    locations2 = -torch.load('./cache/part_sleeves1.pt').flip(-1)
    locations3 = torch.load('./cache/part_sleeves2.pt').flip(-1)

    # locations_ori = locations.clone()

    locations1 *= 5
    locations2 /= 2.5
    locations3 /= 2.5
    locations2[:, 0] += 1.5 * 5
    locations3[:, 0] -= 1.5 * 5

    locations = torch.cat([locations1, locations2, locations3], 0)
    locations_ori = locations.clone()
    return locations, locations_ori

args_optim = {
    'alpha' : [0.1, 0.1],
    'beta' : 10.0,
    'gamma' : 0.5,
    'step_size_max' : 0.1,
    'steps' : 1000,
    'overlap_md' : 1e-3,
    'accumulate_overlap' : True,
}

args_tt = EasyDict({
    'front':{
        'part_ids': [0, 1],
        'if_skip_move': True,
        'init_fun': loc_init_front_back,
    },
    'back':{
        'part_ids': [2, 3],
        'if_skip_move': True,
        'init_fun': loc_init_front_back,
    },
    'sleeves1':{
        'part_ids': [4],
        'init_fun': loc_init_sleeves,
    },
    'sleeves2':{
        'part_ids': [5],
        'init_fun': loc_init_sleeves,

    },
    'fandb':{
        'part_ids': [0, 1, 2, 3],
        'init_fun': loc_init_fandb,

    },
    'tshirt_all':{
        'part_ids': [0, 1, 2, 3, 4, 5],
        'init_fun': loc_init_tshirt_all,

    },
})
print('Done')

mesh_infos = prepare_mesh(mesh_tshirt)

name = 'sleeves1'
locations, locations_ori, info, adj, overlap_set = pipeline(name, args_tt[name], args_optim, mesh_infos, device, retrieve_history=True);

import imageio
images = []

for i in range(10):
    x = info["locations_history"][i]
    fig = show_graph(adj, [x], overlap_set, figsize=[7, 7], axis_equal=True, colors=plt.cm.tab10(4));
    plt.title(f"{i}")
    plt.show()
#     fig.canvas.draw()
#     image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#     image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    image = np.array(fig[0].canvas.buffer_rgba())
    images.append(image)
    plt.close()
imageio.mimsave('./cache/movie.gif', images, fps=3)


