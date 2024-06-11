import torch
import math
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter

class StateVRPTWBase(NamedTuple):
    # fixed input
    coords: torch.Tensor
    demand: torch.Tensor
    ids: torch.Tensor
    time_window: torch.Tensor

    # state
    prev_a: torch.Tensor
    used_capacity: torch.Tensor
    visited_: torch.Tensor
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor
    time: torch.Tensor

class StateVRPTW(StateVRPTWBase):
    VEHICLE_CAPACITY = 1

    @property
    def visited(self):
        if self.visited_.dtype == torch.bool:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.demand.size(-1))

    @property
    def dist(self):
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            return self._replace(
                ids=self.ids[key],
                prev_a=self.prev_a[key],
                used_capacity=self.used_capacity[key],
                visited_=self.visited_[key],
                lengths=self.lengths[key],
                cur_coord=self.cur_coord[key],
                time=self.time[key]
            )
        return super(StateVRPTW, self).__getitem__(key)

    @staticmethod
    def initialize(input, visited_dtype=torch.bool):
        depot = input['depot']
        loc = input['loc']
        demand = input['demand']
        time_window = input['time_window']

        batch_size, n_loc, _ = loc.size()
        return StateVRPTW(
            coords=torch.cat((depot[:, None, :], loc), -2),
            demand=demand,
            time_window=time_window,  # time windows
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            used_capacity=demand.new_zeros(batch_size, 1),
            visited_=(
                torch.zeros(
                    batch_size, 1, n_loc + 1,
                    dtype=torch.bool, device=loc.device
                )
                if visited_dtype == torch.bool
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)
            ),
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=input['depot'][:, None, :],
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),
            time=torch.zeros(batch_size, 1, device=loc.device)
        )

    # def get_final_cost(self):
    #     assert self.all_finished()
    #
    #     return self.lengths + (self.coords[self.ids, 0, :] - self.cur_coord).norm(p=2, dim=-1)

    def update(self, selected):

        assert self.i.size(0) == 1, "can only update if state represents single step"

        selected = selected[:, None]
        prev_a = selected
        n_loc = self.demand.size(-1)

        cur_coord = self.coords[self.ids, selected]

        lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)

        cur_tw = self.time_window[self.ids, selected, :]

        time = self.time + (cur_coord - self.cur_coord).norm(p=2, dim=-1)

        start = torch.max(time, cur_tw[:, :, 0])

        time = start + 10.0 / 60
        # time = start + 10.0 / 80

        time[selected == 0] = 0

        selected_demand = self.demand[self.ids, torch.clamp(prev_a - 1, 0, n_loc - 1)]

        used_capacity = (self.used_capacity + selected_demand) * (prev_a != 0).float()

        if self.visited_.dtype == torch.bool:
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)

        else:
            visited_ = mask_long_scatter(self.visited_, prev_a-1)

        return self._replace(
            prev_a=prev_a, used_capacity=used_capacity, visited_=visited_,
            lengths=lengths, cur_coord=cur_coord, i=torch.Tensor([self.i + 1]),
            time=time
        )

    def all_finished(self):
        return self.i.item() >= self.demand.size(-1) and self.visited.all()

    def get_finished(self):
        return self.visited.sum(-1) == self.visited.size(-1)

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):

        if self.visited_.dtype == torch.bool:
            visited_loc = self.visited_[:, :, 1:]
        else:
            visited_loc = mask_long2bool(self.visited_, n=self.demand.size(-1))

        dis = (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)
        mask_tw = (self.time + dis[self.ids.squeeze(), self.prev_a.squeeze(), :] > self.time_window[:, :, 1]).unsqueeze(1)
        mask_loc = (
            visited_loc |
            (self.demand[self.ids, :] + self.used_capacity[:, :, None] > self.VEHICLE_CAPACITY)
        )

        mask_depot = (self.prev_a == 0) & ((mask_loc == 0).int().sum(-1) > 0)
        return torch.cat((mask_depot[:, :, None], mask_loc), -1) | mask_tw

    def construct_solutions(self, actions):
        return actions