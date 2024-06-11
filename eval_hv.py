#!/usr/bin/env python
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

mt_opt = get_options()
os.environ["CUDA_VISIBLE_DEVICES"] = mt_opt.CUDA_VISIBLE_ID
# print(os.environ["CUDA_VISIBLE_DEVICES"])

# eval_hv_dir = 'sub_outputs/vrptw_50/no_meta_vrptw50_teststep1000_20210321T140630'
# eval_hv_dir = 'sub_outputs/tsp_20/no_meta_tsp20_teststep100_20201106T115713'
# eval_hv_dir = 'sub_outputs/tsp_20/no_meta_tsp20_teststep1000_20201106T115630'
# eval_hv_dir = 'sub_outputs/tsp_20/no_meta_tsp20_teststep5000_20201109T155227'

# eval_hv_dir = 'sub_outputs/tsp_20/reptile_tsp20_ts100_st5_fst10_20201109T154901'
# eval_hv_dir = 'sub_outputs/tsp_20/reptile_tsp20_ts100_st5_fst100_20201109T155014'
# eval_hv_dir = 'sub_outputs/tsp_20/reptile_tsp20_ts100_st5_fst1000_20201109T155030'
# eval_hv_dir = 'sub_outputs/tsp_20/reptile_tsp20_ts100_st5_fst5000_20201109T155336'

# eval_hv_dir = 'transfer_outputs/tsp_20/transfer_step10_20201118T110218'
# eval_hv_dir = 'transfer_outputs/tsp_20/transfer_step100_20201118T110126'
# eval_hv_dir = 'transfer_outputs/tsp_20/transfer_step1000_20201118T110008'
# eval_hv_dir = 'transfer_outputs/tsp_20/transfer_step5000_20201118T110948'

# eval_hv_dir = 'sub_outputs/tsp_50/no_meta_tsp50_teststep10_20201105T221831'
# eval_hv_dir = 'sub_outputs/tsp_50/no_meta_tsp50_teststep100_20201105T221940'
# eval_hv_dir = 'sub_outputs/tsp_50/no_meta_tsp50_teststep1000_20201105T222024'

# eval_hv_dir = 'sub_outputs/tsp_50/reptile_tsp50_ts100_st5_fst10_20201105T221643'
# eval_hv_dir = 'sub_outputs/tsp_50/reptile_tsp50_ts100_st5_fst100_20201105T220611'
# eval_hv_dir = 'sub_outputs/tsp_50/reptile_tsp50_ts100_st5_fst1000_20201105T221447'

# eval_hv_dir = 'transfer_outputs/tsp_50/transfer_step10_20201117T225629'
# eval_hv_dir = 'transfer_outputs/tsp_50/transfer_step100_20201117T224916'
# eval_hv_dir = 'transfer_outputs/tsp_50/transfer_step1000_20201118T100527'

ref = np.array([65., 3.5])
# ref = np.array([1.2, 1.2])
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
    # Initialize baseline
    assert (problem.NAME == 'tsp' or problem.NAME == 'vrptw') and opts.baseline == 'critic', "Critic only supported for TSP or VRPTW"
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
    val_dataset = problem.make_dataset(
        size=opts.graph_size, num_samples=128, filename=opts.val_dataset, distribution=opts.data_distribution)
    w1 = torch.tensor([i * 1.0 / 99 for i in range(100)]).unsqueeze(1)
    # w1 = torch.tensor([i * 1.0 / 9 for i in range(10)]).unsqueeze(1)
    w1 = w1.to(opts.device)
    w2 = 1.0 - w1
    weights = torch.cat((w1, w2), dim=-1)
    cost_objs = []
    for i in range(100):
        weight = weights[i]
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
        lcost_obj1 = []
        lcost_obj2 = []
        for batch in DataLoader(val_dataset, batch_size=opts.eval_batch_size):
            x = move_to(batch, opts.device)
            cost, log_likelihood = model(x)
            lcost_obj1.append(cost[0])
            lcost_obj2.append(cost[1])
        cost_obj = [torch.cat(lcost_obj1, 0).cpu().tolist(), torch.cat(lcost_obj2, 0).cpu().tolist()]
        print("cost_obj:", np.array(cost_obj[0]).mean(), np.array(cost_obj[1]).mean())
        cost_objs.append(cost_obj)
        torch.cuda.empty_cache()
    cost_objs = np.array(cost_objs)
    np.save(eval_hv_dir+".npy", np.mean(cost_objs, axis=2))
    avg_NDS = 0
    avg_hv = 0
    for i in range(cost_objs.shape[2]):
        solutions = cost_objs[:, :, i]
        non_dominated_solutions = get_non_dominated(solutions)
        hv = hypervolume(non_dominated_solutions, ref)
        NDS = non_dominated_solutions.shape[0]
        print("|NDS|=", NDS)
        print("hv=", hv)
        avg_NDS += NDS
        avg_hv += hv
    avg_NDS /= cost_objs.shape[2]
    avg_hv /= cost_objs.shape[2]
    print("avg_|NDS|=", avg_NDS)
    print("avg_hv=", avg_hv)

    print("end_date: " + time.strftime("%Y%m%dT%H%M%S"))
    print("eval_hv_dir:", eval_hv_dir)
if __name__ == "__main__":
    run(mt_opt)
