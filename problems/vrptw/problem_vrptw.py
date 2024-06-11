from torch.utils.data import Dataset
import torch
import os
import pickle
import numpy as np
import math

from problems.vrptw.state_vrptw import StateVRPTW


class VRPTW(object):

    NAME = 'vrptw'  # vehicle routing problem with time window

    VEHICLE_CAPACITY = 1.0  # demands should be normalized

    MAX1 = 30
    MAX2 = 3
    MIN1 = 5
    MIN2 = 1

    # VEHICLE_CAPACITY = 200

    @staticmethod
    def get_costs(dataset, pi):
        # dataset['depot'] = dataset['depot'] * 60
        # dataset['demand'] = dataset['demand'] * 200
        # dataset['loc'] = dataset['loc'] * 60
        # dataset['time_window'] = dataset['time_window'] * 60
        down = 60
        # down = 80
        batch_size, graph_size = dataset['demand'].size()

        depot_pos = dataset['depot']  # batch_size * 2
        loc_pos = dataset['loc']  # batch_size * graph_size * 2
        loc = torch.cat((depot_pos[:, None, :], loc_pos), dim=1)
        # loc = loc * down
        # loc = loc * 100
        dis = (loc[:, :, None, :] - loc[:, None, :, :]).norm(p=2, dim=-1)
        # batch_size * (graph_size + 1) * (graph_size + 1)

        # tw = dataset['time_window']  # batch_size * (graph_size + 1) * 2
        tw = dataset['time_window']  # * down  # batch_size * (graph_size + 1) * 2
        demands = torch.cat(
            (
                torch.full_like(dataset['demand'][:, :1], -VRPTW.VEHICLE_CAPACITY),
                dataset['demand']
            ),
            1
        )
        rng = torch.arange(batch_size, out=demands.data.new().long())
        used_cap = torch.zeros_like(dataset['demand'][:, 0])
        used_time = torch.zeros_like(dataset['demand'][:, 0])
        used_dis = torch.zeros_like(dataset['demand'][:, 0])
        max_len = torch.zeros_like(dataset['demand'][:, 0])
        a_prev = None
        begin = torch.zeros_like(dataset['demand'][:, 0], dtype=torch.long)
        for a in pi.transpose(0, 1):
            assert a_prev is None or (demands[((a_prev == 0) & (a == 0)), :] < 1e-7).all(), \
                "can not visit depot twice if any nonzero demand"
            d = torch.min(demands[rng, a], VRPTW.VEHICLE_CAPACITY - used_cap)
            demands[rng, a] -= d
            used_cap += d
            used_cap[a == 0] = 0
            a_prev = a
            # check the time window
            # used_dis[a == 0] = 0
            used_time += dis[rng, begin, a]
            used_dis += dis[rng, begin, a]
            begin = a
            max_len[used_dis > max_len] = used_dis[used_dis > max_len]
            # used_time[a == 0] = 0
            assert (used_time <= tw[rng, a, 1]).all(), \
                "must respect time window constraints"
            used_time += 10.0 / 60
            # used_time += 10.0
            used_time[a == 0] = 0
            used_dis[a == 0] = 0
        assert (demands < 1e-7).all(),  "all demand must be satisfied"
        """code must be added here"""

        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        loc_with_depot = loc_with_depot  # * down
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))
        f1 = (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
            + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)
            + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)
        )
        f2 = max_len
        """
        min1 = torch.min(f1)
        min2 = torch.min(f2)
        max1 = torch.max(f1)
        max2 = torch.max(f2)
        if min1 < VRPTW.MIN1:
            VRPTW.MIN1 = min1
        if min2 < VRPTW.MIN2:
            VRPTW.MIN2 = min2
        if max1 > VRPTW.MAX1:
            VRPTW.MAX1 = max1
        if max2 > VRPTW.MAX2:
            VRPTW.MAX2 = max2
        nor_f1 = (f1 - VRPTW.MIN1) / (VRPTW.MAX1 - VRPTW.MIN1)
        nor_f2 = (f2 - VRPTW.MIN2) / (VRPTW.MAX2 - VRPTW.MIN2)
        """
        return [f1, f2], None  # , [nor_f1, nor_f2] normalization should not be here
        # return (
        #     (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
        #     + (d[:, 0] - dataset['depot'].norm(p=2, dim=1))
        #     + (d[:, -1]) - dataset['depot'].norm(p=2, dim=1)
        # ), max_len

    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPTWDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateVRPTW.initialize(*args, **kwargs)


def make_instance(args):
    depot, loc, demand, capacity, time_window, *args = args
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, grid_size = args
    return {
        'loc': torch.Tensor(loc, dtype=torch.float) / grid_size,
        'demand': torch.Tensor(demand, dtype=torch.float) / capacity,
        'depot': torch.Tensor(depot, dtype=torch.float) / grid_size,
        'time_window': torch.Tensor(time_window, dtype=torch.float)/grid_size
    }


class VRPTWDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(VRPTWDataset, self).__init__()
        self.dataset = []
        self.num_samples = num_samples
        self.size = size
        #  ff = 'vrptw-' + str(self.size) + '-data-' + str(self.num_samples) + '.pkl'
        ff = None
        if num_samples <= 100000:
            ff = 'vrptw-' + str(self.size) + '-data-' + str(100000) + '.pkl'
        elif num_samples <= 1000000:
            ff = 'vrptw-' + str(self.size) + '-data-' + str(1000000) + '.pkl'
        elif num_samples <= 10000:
            ff = 'vrptw-' + str(self.size) + '-data-' + str(10000) + '.pkl'
        elif num_samples <= 2560000:
            ff = 'vrptw-' + str(self.size) + '-data-' + str(2560000) + '.pkl'
        if ff is not None and os.path.exists(ff):
            filename = ff
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = data[:num_samples]
        else:
            self.generate_data(size=size, num_samples=num_samples, offset=offset, distribution=None)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def generate_data(self, filename=None,size=50, num_samples=1000000, offset=0, distribution=None):
        """
            CAPACITIES = {
                        10: 20.,
                        20: 30.,
                        50: 40.,
                        100: 50.
                    }
        """
        CAPACITIES = 200
        # CAPACITIES = 1000
        e0 = 0  # start time of depot
        # l0 = 230.0 / 60  # end time of depot
        # l0 = 1000.0 / 80
        l0 = 230.0 / 60
        service_time = 10 / 60
        # service_time = 10 / 80
        self.data = [
            {
                'loc': torch.FloatTensor(size, 2).uniform_(0, 1),
                # 'demand': (torch.FloatTensor(size).uniform_(0, 30).int() + 1).float() / CAPACITIES,
                'demand': (torch.FloatTensor(size).uniform_(0, 42).int() + 1).float() / CAPACITIES,
                'depot': torch.FloatTensor(2).uniform_(0, 1),
                'time_window': torch.FloatTensor(size, 2).uniform_(0, 5)  # add time window
            }
            for i in range(num_samples)
        ]

        low = 0.1
        high = 3
        # high = 7
        # high = 2.3
        for i in range(num_samples):
            depot_pos = self.data[i]['depot']
            loc_pos = self.data[i]['loc']
            depot_pos = depot_pos.unsqueeze(0).repeat(size, 1)
            dis = (depot_pos - loc_pos).norm(p=2, dim=-1)
            start = e0 + dis
            end = l0 - dis - service_time
            center = np.random.uniform(start, end)
            # rand = (np.random.randn(size) + mean) * std
            width = np.random.uniform(low, high, size)
            e = torch.Tensor(center - width / 2)
            l = torch.Tensor(center + width / 2)
            e[e < 0] = 0
            l[l > end] = end[l > end] - 1e-5
            e_0 = torch.FloatTensor([e0])
            l_0 = torch.FloatTensor([l0])
            e = torch.cat((e_0, e))
            e = e.unsqueeze(1)
            l = torch.cat((l_0, l))
            l = l.unsqueeze(1)
            tw = torch.cat((e, l), dim=-1)
            self.data[i]['time_window'] = tw
        filename = 'vrptw-' + str(self.size) + '-data-' + str(self.num_samples) + '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self.data, f, pickle.HIGHEST_PROTOCOL)
            print(filename + 'is saved!')




if __name__ == '__main__':
    VRPTWDataset(num_samples=100000)