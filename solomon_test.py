import re
import os
import numpy as np
# import matplotlib.pyplot as plt
import json
import pprint as pp
import random
import torch
from torch import nn
import torch.optim as optim
import time
from nets.attention_model import set_decode_type
from tqdm import tqdm
# from tensorboard_logger import Logger as TbLogger

from nets.critic_network import CriticNetwork
from options import get_options
from train import train_epoch, validate, get_inner_model
from reinforce_baselines import CriticBaseline
from nets.attention_model import AttentionModel
from utils import torch_load_cpu, load_problem
from torch.utils.data import DataLoader
from utils import move_to
from copy import deepcopy
from train import clip_grad_norms
from collections import OrderedDict
from non_dominated_sort import get_non_dominated
from hypervolume import hypervolume
# import matplotlib.pyplot as plt

mt_opt = get_options()
os.environ["CUDA_VISIBLE_DEVICES"] = mt_opt.CUDA_VISIBLE_ID

runtime = 0.0
def norm(a):
    amin, amax = a.min(), a.max()
    b = (a - amin) / (amax - amin)
    return b


# dataset = TSPDataset(filename=test_dataset)
# res = validate(model, dataset, opts)
eval_str = mt_opt.eval_hv_dir
size_str = str(mt_opt.graph_size)
ins_name = mt_opt.ins_name
if eval_str[0:8] == 'transfer':
    m_type = 'transfer'
elif eval_str.find('reptile')!=-1:
    m_type = 'reptile'
elif eval_str.find('no_meta')!=-1:
    m_type = 'rand'
# npy_save_path = 'kroAB100_tranfer_step1000'
ft_step = int(re.search(r'(step|fst)(.*?)_202', eval_str, re.M | re.I).group(2))

npy_save_path = 'solomon' + size_str + '_' + ins_name + '_' + m_type + '_' + 'step'+ str(ft_step)
if m_type == 'transfer':
    if mt_opt.transfer_obj2 is False:
        npy_save_path = 'solomon' + size_str + '_' + ins_name + '_' + m_type + '_obj1_' + 'step' + str(ft_step)
    else:
        npy_save_path = 'solomon' + size_str + '_' + ins_name + '_' + m_type + '_obj2_' + 'step' + str(ft_step)

if mt_opt.is_WB is True:
    npy_save_path = npy_save_path + '_WB'
npy_save_path = npy_save_path + '.npy'
npy_save_path = 'solomon_res_npy/' + npy_save_path

data_npy = np.loadtxt('solomon_' + size_str + '/' + ins_name + '.txt', skiprows=9)
# size = 25
down = 60
CAPACITIES = 200
# CAPACITIES = 1000
batch = 10
e0 = 0  # start time of depot
l0 = 230.0 / down
# l0 = 1000.0 / 60
# service_time = 10.0 / (math.sqrt(2) * 60)
data = {
            'loc': torch.FloatTensor((data_npy[1:, 1:3] - 0) * 1.0 / down).unsqueeze(0),
            'demand': torch.FloatTensor(data_npy[1:, 3]).unsqueeze(0).float() / CAPACITIES,
            'depot': torch.FloatTensor((data_npy[0, 1:3] - 0) * 1.0 / down).unsqueeze(0),
            'time_window': torch.FloatTensor(data_npy[:, 4:6] * 1.0 / down).unsqueeze(0)
            # add time window
    }
# data = {
#             'loc': torch.FloatTensor((data_npy[1:, 1:3] - 0) * 1.0 / down).unsqueeze(0).repeat(batch, 1, 1),
#             'demand': torch.FloatTensor(data_npy[1:, 3]).unsqueeze(0).float() / CAPACITIES,
#             'depot': torch.FloatTensor((data_npy[0, 1:3] - 0) * 1.0 / down).unsqueeze(0).repeat(batch, 1),
#             'time_window': torch.FloatTensor(data_npy[:, 4:6] * 1.0 / down).unsqueeze(0).repeat(batch, 1, 1)
#             # add time window
#     }

# data['loc'] = torch.FloatTensor((a[1:, 1:3] - 5) * 1.0 / 60).unsqueeze(0).repeat(batch, 1, 1)
# data['demand'] = torch.FloatTensor(a[1:, 3]).unsqueeze(0).float() / CAPACITIES

# data['demand'] = data['demand'].repeat(batch, 1)

# data['depot'] = torch.FloatTensor((a[0, 1:3] - 5) * 1.0 / 60).unsqueeze(0).repeat(batch, 1)
# data['time_window'] = torch.FloatTensor(a[:, 4:6] * 1.0 / 60).unsqueeze(0).repeat(batch, 1, 1)
ref = np.array([65., 3.5])
bat = data

def run(opts):
    eval_hv_dir = opts.eval_hv_dir
    opts.baseline = 'critic'
    pp.pprint(vars(opts))

    print("run_date: " + time.strftime("%Y%m%dT%H%M%S"))
    # Set the random seed
    torch.manual_seed(opts.seed)

    # Set the device
    # opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")
    opts.device = torch.device("cuda")
    problem = load_problem(opts.problem)

    model_class = {
        'attention': AttentionModel,
    }.get(opts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    model = model_class(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size
    ).to(opts.device)

    if opts.use_cuda and torch.cuda.device_count() > 1 and len(mt_opt.CUDA_VISIBLE_ID) > 1:
        model = torch.nn.DataParallel(model)
    # Initialize baseline
    assert problem.NAME == 'vrptw' and opts.baseline == 'critic', "Critic only supported for VRPTW"
    baseline = CriticBaseline(
        (
            CriticNetwork(
                5,
                opts.embedding_dim,
                opts.hidden_dim,
                opts.n_encode_layers,
                opts.normalization
            )
        ).to(opts.device)
    )
    # if opts.bl_warmup_epochs > 0:
    #    baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': mt_opt.meta_lr}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': mt_opt.meta_lr}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    # w1 = torch.tensor([i * 1.0 / 99 for i in range(100)]).unsqueeze(1)
    # # w1 = torch.tensor([i * 1.0 / 9 for i in range(10)]).unsqueeze(1)
    # w1 = w1.to(opts.device)
    # w2 = 1.0 - w1
    # weights = torch.cat((w1, w2), dim=-1)
    cost_objs = []

    # weight = torch.FloatTensor([0, 1])

    pis = []

    for i in range(100):
        # weight = weights[i]
        ################################################
        load_path = eval_hv_dir + '/model-{}.pt'.format(i)
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)
        model_ = get_inner_model(model)
        if opts.is_load_multi:
            state_dict = load_data.get('model', {})
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            model_.load_state_dict({**model_.state_dict(), **new_state_dict})
        else:
            model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})
        if 'baseline' in load_data:
            baseline.load_state_dict(load_data['baseline'])
        if 'optimizer' in load_data:
            optimizer.load_state_dict(load_data['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    # if isinstance(v, torch.Tensor):
                    if torch.is_tensor(v):
                        state[k] = v.to(opts.device)
        ################################################

        set_decode_type(model, "greedy")

        x = move_to(bat, opts.device)
        t1 = time.time()
        cost, log_likelihood, pi = model(x, return_pi=True)
        t2 = time.time()
        global runtime
        runtime = runtime + (t2-t1)
        print("delta_time:", t2-t1)
        print("cost_obj:", cost[0], cost[1])
        pis.append(pi.squeeze(0))
        cost_objs.append([cost[0].cpu().tolist()[0], cost[1].cpu().tolist()[0]])
        torch.cuda.empty_cache()
    cost_objs = np.array(cost_objs)
    # pis = torch.stack(pis)
    # pis = pis.cpu().numpy()

    non_dominated_solutions = get_non_dominated(cost_objs)

    hv = hypervolume(non_dominated_solutions, ref)
    NDS = non_dominated_solutions.shape[0]
    print("|NDS|=", NDS)
    print("hv=", hv)

    print("end_date: " + time.strftime("%Y%m%dT%H%M%S"))
    print("eval_hv_dir:", eval_hv_dir)

    return non_dominated_solutions


if __name__ == "__main__":
    res = run(mt_opt)
    np.save(npy_save_path, res)
    print("npy_save_path:", npy_save_path)
    print("runtime:", round(runtime,2))
    # plt.figure()
    # plt.plot(res[:, 0], res[:, 1], 'r+')
    # plt.xlabel('f1')
    # plt.ylabel('f2')
    # plt.title('MOTSP')
    # plt.show()
