
import pymzn
import numpy as np
import networkx as nx

from .problem import Problem
from .utils import subdict, freeze, _phi, _infer, _improve


PROBLEM = r"""

int: N_ROOMS;
set of int: ROOMS = 1..N_ROOMS;
array[ROOMS] of int: dist_bathroom;
array[ROOMS] of int: dist_restaurant;

int: max_normal_bathroom_dist;
int: max_suite_bathroom_dist;
int: max_suite_restaurant_dist;

array[ROOMS] of int: capacity;
int: max_capacity = max(capacity);

array[ROOMS] of var 0..max_capacity: single_beds;
array[ROOMS] of var 0..max_capacity: double_beds;
array[ROOMS] of var 0..max_capacity: bunk_beds;
array[ROOMS] of var 0..max_capacity: tables;
array[ROOMS] of var 0..max_capacity: sofas;

array[ROOMS] of var int: beds = [single_beds[room] + double_beds[room] + bunk_beds[room] | room in ROOMS];
array[ROOMS] of var int: furniture = [beds[room] + tables[room] + sofas[room] | room in ROOMS];

%-------------------------------------------------------------------------------%
% ROOM TYPES
%-------------------------------------------------------------------------------%

array[ROOMS] of var bool: dorm = [not (normal[room] \/ suite[room]) | room in ROOMS];

array[ROOMS] of var bool: normal = [
           beds[room] <= 3
        /\ bunk_beds[room] == 0
        /\ dist_bathroom[room] <= max_normal_bathroom_dist
        /\ tables[room] >= 1
    | room in ROOMS];

array[ROOMS] of var bool: suite = [
           beds[room] <= 1
        /\ bunk_beds[room] == 0
        /\ dist_bathroom[room] <= max_suite_bathroom_dist
        /\ tables[room] >= 1
        /\ sofas[room] >= 1
    | room in ROOMS];

var int: n_dorm = sum(dorm);
var int: n_normal = sum(normal);
var int: n_suite = sum(suite);


%-------------------------------------------------------------------------------%
% FEATURES
%-------------------------------------------------------------------------------%

% TYPE PERCENTAGE

int: dorm_th1;
int: dorm_th2;
int: dorm_th3;
int: normal_th1;
int: normal_th2;
int: normal_th3;
int: suite_th1;
int: suite_th2;
int: suite_th3;

var bool: dorm_below_th1 = n_dorm <= dorm_th1;
var bool: dorm_below_th2 = n_dorm <= dorm_th2;
var bool: dorm_below_th3 = n_dorm <= dorm_th3;
var bool: dorm_above_th3 = n_dorm > dorm_th3;

var bool: normal_below_th1 = n_normal <= normal_th1;
var bool: normal_below_th2 = n_normal <= normal_th2;
var bool: normal_below_th3 = n_normal <= normal_th3;
var bool: normal_above_th3 = n_normal > normal_th3;

var bool: suite_below_th1 = n_suite <= suite_th1;
var bool: suite_below_th2 = n_suite <= suite_th2;
var bool: suite_below_th3 = n_suite <= suite_th3;
var bool: suite_above_th3 = n_suite > suite_th3;


% COST

int: single_bed_cost;
int: double_bed_cost;
int: bunk_bed_cost;
int: table_cost;
int: sofa_cost;

var int: total_cost = sum(room in ROOMS)(
      single_bed_cost * single_beds[room]
    + double_bed_cost * double_beds[room]
    + bunk_bed_cost * bunk_beds[room]
    + table_cost * tables[room]
    + sofa_cost * sofas[room]
);

int: cost_th1;
int: cost_th2;
int: cost_th3;

var bool: cost_below_th1 = total_cost <= cost_th1;
var bool: cost_below_th2 = total_cost <= cost_th2;
var bool: cost_below_th3 = total_cost <= cost_th3;
var bool: cost_above_th3 = total_cost > cost_th3;

% GUESTS

array[ROOMS] of var int: guests = [single_beds[room] + 2 * double_beds[room] + 3 * bunk_beds[room] | room in ROOMS];
var int: total_guests = sum(guests);

int: guests_th1;
int: guests_th2;
int: guests_th3;

var bool: guests_below_th1 = total_guests <= guests_th1;
var bool: guests_below_th2 = total_guests <= guests_th2;
var bool: guests_below_th3 = total_guests <= guests_th3;
var bool: guests_above_th3 = total_guests > guests_th3;

% FEATURES

int: room_furniture_th1;
int: room_furniture_th2;
int: room_furniture_th3;

int: N_FEATURES = 20 + 7 * N_ROOMS;
set of int: FEATURES = 1..N_FEATURES;
"""

FULL = """
array[FEATURES] of var int: phi = [
        2 * dorm_below_th1 - 1,
        2 * dorm_below_th2 - 1,
        2 * dorm_below_th3 - 1,
        2 * dorm_above_th3 - 1,
        2 * normal_below_th1 - 1,
        2 * normal_below_th2 - 1,
        2 * normal_below_th3 - 1,
        2 * normal_above_th3 - 1,
        2 * suite_below_th1 - 1,
        2 * suite_below_th2 - 1,
        2 * suite_below_th3 - 1,
        2 * suite_above_th3 - 1,
        2 * cost_below_th1 - 1,
        2 * cost_below_th2 - 1,
        2 * cost_below_th3 - 1,
        2 * cost_above_th3 - 1,
        2 * guests_below_th1 - 1,
        2 * guests_below_th2 - 1,
        2 * guests_below_th3 -1,
        2 * guests_above_th3 -1]
    ++ [2 * dorm[room] - 1 | room in ROOMS]
    ++ [2 * normal[room] - 1 | room in ROOMS]
    ++ [2 * suite[room] - 1 | room in ROOMS]
    ++ [2 * (furniture[room] <= room_furniture_th1) - 1 | room in ROOMS]
    ++ [2 * (furniture[room] <= room_furniture_th2) - 1 | room in ROOMS]
    ++ [2 * (furniture[room] <= room_furniture_th3) - 1 | room in ROOMS]
    ++ [2 * (furniture[room] > room_furniture_th3) - 1 | room in ROOMS];
"""


class Hotel(nx.Graph):

    def __init__(self, max_normal_bathroom_dist=2, max_suite_bathroom_dist=1,
                 max_suite_restaurant_dist=2, dorm_th1=4, dorm_th2=8,
                 dorm_th3=12, normal_th1=4, normal_th2=8, normal_th3=12,
                 suite_th1=4, suite_th2=8, suite_th3=12, cost_th1=100,
                 cost_th2=200, cost_th3=300, guests_th1=20, guests_th2=40,
                 guests_th3=60, room_furniture_th1=2, room_furniture_th2=4,
                 room_furniture_th3=6, single_bed_cost=10, double_bed_cost=20,
                 bunk_bed_cost=15, table_cost=15, sofa_cost=25):
        self.args = subdict(locals(), nokeys=['self', '__class__'])
        super().__init__()
        self.rooms = []
        self.capacity = []
        self.restaurants = []
        self.bathrooms = []
        self._shortest_paths = None
        self._room_bathroom_dists = None
        self._room_restaurant_dists = None

    @classmethod
    def default(cls):
        h = cls()
        h.add_restaurant(1), h.add_restaurant(2)
        h.add_bathroom(3), h.add_bathroom(4)
        h.add_bathroom(5), h.add_bathroom(6)
        h.add_room(7, 6), h.add_room(8, 6), h.add_room(10, 3),
        h.add_room(11, 3), h.add_room(12, 3), h.add_room(13, 6),
        h.add_room(14, 3), h.add_room(15, 6), h.add_room(16, 3),
        h.add_room(17, 3), h.add_room(18, 3), h.add_room(19, 4),
        h.add_room(20, 3), h.add_room(21, 4), h.add_room(22, 6)

        h.add_edge(1, 2), h.add_edge(1, 17)
        h.add_edge(2, 1), h.add_edge(2, 7), h.add_edge(2, 18)
        h.add_edge(7, 2), h.add_edge(7, 19), h.add_edge(7, 5), h.add_edge(7, 8)
        h.add_edge(8, 7), h.add_edge(8, 5)
        h.add_edge(3, 20), h.add_edge(3, 4), h.add_edge(3, 10)
        h.add_edge(4, 3), h.add_edge(4, 20), h.add_edge(4, 10)
        h.add_edge(10, 20), h.add_edge(10, 4), h.add_edge(10, 11)
        h.add_edge(11, 21), h.add_edge(11, 10), h.add_edge(11, 12)
        h.add_edge(12, 6), h.add_edge(12, 11), h.add_edge(12, 13)
        h.add_edge(13, 6), h.add_edge(13, 12), h.add_edge(13, 14)
        h.add_edge(14, 6), h.add_edge(14, 13), h.add_edge(14, 15)
        h.add_edge(15, 16), h.add_edge(15, 22), h.add_edge(15, 6)
        h.add_edge(15, 14)
        h.add_edge(16, 17), h.add_edge(16, 22), h.add_edge(16, 15)
        h.add_edge(17, 1), h.add_edge(17, 18), h.add_edge(17, 16)
        h.add_edge(18, 2), h.add_edge(18, 19), h.add_edge(18, 22)
        h.add_edge(18, 17)
        h.add_edge(19, 7), h.add_edge(19, 5), h.add_edge(19, 18)
        h.add_edge(5, 7), h.add_edge(5, 8), h.add_edge(5, 19)
        h.add_edge(20, 3), h.add_edge(20, 4), h.add_edge(20, 10)
        h.add_edge(20, 21)
        h.add_edge(21, 20), h.add_edge(21, 6), h.add_edge(21, 11)
        h.add_edge(6, 21), h.add_edge(6, 22), h.add_edge(6, 12)
        h.add_edge(6, 13), h.add_edge(6, 14), h.add_edge(6, 15)
        h.add_edge(22, 18), h.add_edge(22, 6), h.add_edge(22, 15)
        h.add_edge(22, 16)
        return h

    @classmethod
    def default2(cls):
        h = cls()
        h.add_restaurant(1)
        h.add_bathroom(2), h.add_bathroom(3)
        h.add_room(4, 3), h.add_room(5, 3), h.add_room(6, 3), h.add_room(7, 3),
        h.add_room(8, 6), h.add_room(9, 8), h.add_room(10, 8), h.add_room(11, 6)
        h.add_room(12, 4), h.add_room(13, 4)

        h.add_edge(1, 5), h.add_edge(1, 6), h.add_edge(1, 8), h.add_edge(1, 11)
        h.add_edge(2, 4), h.add_edge(2, 11)
        h.add_edge(3, 7), h.add_edge(3, 8)
        h.add_edge(4, 2), h.add_edge(4, 5), h.add_edge(4, 11)
        h.add_edge(5, 4), h.add_edge(5, 1), h.add_edge(5, 11)
        h.add_edge(6, 1), h.add_edge(6, 7), h.add_edge(6, 8)
        h.add_edge(7, 6), h.add_edge(7, 3), h.add_edge(7, 8)
        h.add_edge(8, 1), h.add_edge(8, 6), h.add_edge(8, 7), h.add_edge(8, 3)
        h.add_edge(8, 9), h.add_edge(8, 11)
        h.add_edge(9, 8), h.add_edge(9, 10), h.add_edge(9, 13)
        h.add_edge(10, 9), h.add_edge(10, 11), h.add_edge(10, 12)
        h.add_edge(11, 10), h.add_edge(11, 8), h.add_edge(11, 2), h.add_edge(11, 4)
        h.add_edge(11, 5), h.add_edge(11, 1)
        h.add_edge(12, 10), h.add_edge(12, 13)
        h.add_edge(13, 12), h.add_edge(13, 9)
        return h

    def add_room(self, _id, capacity):
        self.add_node(_id, _type='room')
        self.rooms.append(_id)
        self.capacity.append(capacity)

    def add_restaurant(self, _id):
        self.add_node(_id, _type='restaurant')
        self.restaurants.append(_id)

    def add_bathroom(self, _id):
        self.add_node(_id, _type='bathroom')
        self.bathrooms.append(_id)

    @property
    def num_rooms(self):
        return len(self.rooms)

    @property
    def room_bathroom_dists(self):
        if self._room_bathroom_dists is not None:
            return self._room_bathroom_dists

        if self._shortest_paths is None:
            self._shortest_paths = nx.all_pairs_shortest_path_length(self)

        room_bathroom_dists = []
        for room in self.rooms:
            room_bathroom_dist = min([self._shortest_paths[room][bathroom]
                                      for bathroom in self.bathrooms])
            room_bathroom_dists.append(room_bathroom_dist)

        self._room_bathroom_dists = room_bathroom_dists
        return room_bathroom_dists

    @property
    def room_restaurant_dists(self):
        if self._room_restaurant_dists is not None:
            return self._room_restaurant_dists

        if self._shortest_paths is None:
            self._shortest_paths = nx.all_pairs_shortest_path_length(self)

        room_restaurant_dists = []
        for room in self.rooms:
            room_restaurant_dist = min([self._shortest_paths[room][restaurant]
                                      for restaurant in self.restaurants])
            room_restaurant_dists.append(room_restaurant_dist)

        self._room_restaurant_dists = room_restaurant_dists
        return room_restaurant_dists

    @property
    def adjacency_matrix(self):
        return np.array(nx.adjacency_matrix(self, nodelist=self.rooms).todense())


class HotelsProblem:

    def __init__(self, *args, hotel=Hotel.default()):
        self.num_rooms = hotel.num_rooms
        self.args = {'N_ROOMS': hotel.num_rooms,
                     'capacity': hotel.capacity,
                     'dist_bathroom': hotel.room_bathroom_dists,
                     'dist_restaurant': hotel.room_restaurant_dists,
                     **hotel.args}
        self.n_features = 20 + 7 * hotel.num_rooms
        self.parts = list(range(hotel.num_rooms))
        self.size = [20] * (hotel.num_rooms - 1) + [0]
        self.phis = {}
        #self.solver = pymzn.oscar_cbls
        self.solver = pymzn.gecode
        self.keep = False

    def initial_configuration(self):
        """Returns an initial configuration.

        Current implementation simply solves a phi (solve satisfy) problem with
        no input object, and thus generating a random object.
        """
        return pymzn.minizinc(_phi(PROBLEM + FULL), data={**self.args}, keep=self.keep, force_flatten=True)[0]

    def phi(self, x):
        _frx = freeze(x)
        if _frx in self.phis:
            return self.phis[_frx]
        phi = pymzn.minizinc(_phi(PROBLEM + FULL), data={**self.args, **x},
                             output_vars=['phi'], keep=self.keep, force_flatten=True)[0]['phi']
        self.phis[_frx] = np.array(phi)
        return self.phis[_frx]

    def partial_utility(self, x, w, part):
        """Returns the partial utility of an object.

        The partial utility is the utility calculated over all the features of
        the part.

        Parameters
        ----------
        x : obj
            The input object.
        w : np.ndarray
            The weights used to calculate the partial utility.
        part : int
            The index of the part which to calculate the partial utility on.
        """
        _phi = self.phi(x)
        p_w = np.zeros(w.shape)
        p_w[:20] = w[:20]
        p_w[20 + 0 * len(self.parts) + part] = w[20 + 0 * len(self.parts) + part]
        p_w[20 + 1 * len(self.parts) + part] = w[20 + 1 * len(self.parts) + part]
        p_w[20 + 2 * len(self.parts) + part] = w[20 + 2 * len(self.parts) + part]
        p_w[20 + 3 * len(self.parts) + part] = w[20 + 3 * len(self.parts) + part]
        p_w[20 + 4 * len(self.parts) + part] = w[20 + 4 * len(self.parts) + part]
        p_w[20 + 5 * len(self.parts) + part] = w[20 + 5 * len(self.parts) + part]
        p_w[20 + 6 * len(self.parts) + part] = w[20 + 6 * len(self.parts) + part]
        return p_w.dot(_phi)

    def local_update(self, w, x, xbar, part, eta=1.0):
        """Returns the update (delta) over the local subutility of the part.

        The local subutility is *not* the partial utility. The local subutility
        is the one used to make inference, including only the features of the
        part that do not depend on the following parts. This function updates
        only the features that belong to the local subutility of the given part,
        ignoring all the feature excluded because belonging to following parts.

        Parameters
        ----------
        w : np.ndarray
            The current weights.
        x : obj
            The input object.
        xbar : obj
            The improved object.
        part : int
            The index of the part whose local subutility has to be updated.
        """
        phibar = self.phi(xbar)
        phi = self.phi(x)
        p_w = np.array(w)
        if part == len(self.parts) - 1:
            p_w[0:20] = w[0:20] + eta * (phibar[0:20] - phi[0:20])
        p_w[20 + 0 * len(self.parts) + part] = w[20 + 0 * len(self.parts) + part] + eta * (phibar[20 + 0 * len(self.parts) + part] - phi[20 + 0 * len(self.parts) + part])
        p_w[20 + 1 * len(self.parts) + part] = w[20 + 1 * len(self.parts) + part] + eta * (phibar[20 + 1 * len(self.parts) + part] - phi[20 + 1 * len(self.parts) + part])
        p_w[20 + 2 * len(self.parts) + part] = w[20 + 2 * len(self.parts) + part] + eta * (phibar[20 + 2 * len(self.parts) + part] - phi[20 + 2 * len(self.parts) + part])
        p_w[20 + 3 * len(self.parts) + part] = w[20 + 3 * len(self.parts) + part] + eta * (phibar[20 + 3 * len(self.parts) + part] - phi[20 + 3 * len(self.parts) + part])
        p_w[20 + 4 * len(self.parts) + part] = w[20 + 4 * len(self.parts) + part] + eta * (phibar[20 + 4 * len(self.parts) + part] - phi[20 + 4 * len(self.parts) + part])
        p_w[20 + 5 * len(self.parts) + part] = w[20 + 5 * len(self.parts) + part] + eta * (phibar[20 + 5 * len(self.parts) + part] - phi[20 + 5 * len(self.parts) + part])
        p_w[20 + 6 * len(self.parts) + part] = w[20 + 6 * len(self.parts) + part] + eta * (phibar[20 + 6 * len(self.parts) + part] - phi[20 + 6 * len(self.parts) + part])
        return p_w

    def local_subutility(self, x, w, part):
        """Returns the local subutility of an object.

        The local subutility is the utility calculated over the features of
        the part that do not belong to the following parts.

        Parameters
        ----------
        x : obj
            The input object.
        w : np.ndarray
            The weights used to calculate the local subutility.
        part : int
            The index of the part which to calculate the local subutility on.
        """
        _phi = self.phi(x)
        p_w = np.zeros(w.shape)
        if part == len(self.parts) - 1:
            p_w[:20] = w[:20]
        p_w[20 + 0 * len(self.parts) + part] = w[20 + 0 * len(self.parts) + part]
        p_w[20 + 1 * len(self.parts) + part] = w[20 + 1 * len(self.parts) + part]
        p_w[20 + 2 * len(self.parts) + part] = w[20 + 2 * len(self.parts) + part]
        p_w[20 + 3 * len(self.parts) + part] = w[20 + 3 * len(self.parts) + part]
        p_w[20 + 4 * len(self.parts) + part] = w[20 + 4 * len(self.parts) + part]
        p_w[20 + 5 * len(self.parts) + part] = w[20 + 5 * len(self.parts) + part]
        p_w[20 + 6 * len(self.parts) + part] = w[20 + 6 * len(self.parts) + part]
        return p_w.dot(_phi)

    def _local_subutility(self, model, part):
        """Internal function to calculate the local subutility for local
        inference. Not used outside of this class.
        """

        if part == len(self.parts) - 1:
            features = ['2 * dorm_below_th1 - 1',
                        '2 * dorm_below_th2 - 1',
                        '2 * dorm_below_th3 - 1',
                        '2 * dorm_above_th3 - 1',
                        '2 * normal_below_th1 - 1',
                        '2 * normal_below_th2 - 1',
                        '2 * normal_below_th3 - 1',
                        '2 * normal_above_th3 - 1',
                        '2 * suite_below_th1 - 1',
                        '2 * suite_below_th2 - 1',
                        '2 * suite_below_th3 - 1',
                        '2 * suite_above_th3 - 1',
                        '2 * cost_below_th1 - 1',
                        '2 * cost_below_th2 - 1',
                        '2 * cost_below_th3 - 1',
                        '2 * cost_above_th3 - 1',
                        '2 * guests_below_th1 - 1',
                        '2 * guests_below_th2 - 1',
                        '2 * guests_below_th3 - 1',
                        '2 * guests_above_th3 - 1']
        else:
            features = [0] * 20

        room_feats = [
            '2 * dorm[{0}] - 1',
            '2 * normal[{0}] - 1',
            '2 * suite[{0}] - 1',
            '2 * (furniture[{0}] <= room_furniture_th1) - 1',
            '2 * (furniture[{0}] <= room_furniture_th2) - 1',
            '2 * (furniture[{0}] <= room_furniture_th3) - 1',
            '2 * (furniture[{0}] > room_furniture_th3) - 1']

        for feat in room_feats:
            for p in self.parts:
                if p == part:
                    features.append(feat.format(part + 1))
                else:
                    features.append(0)
        features = np.array(features)
        model.array_variable('phi', 'FEATURES', 'int', features)
        return model

    def _partial_model(self, problem, part, x):
        model = pymzn.MiniZincModel(problem)
        for i in range(self.num_rooms):
            if i != part:
                room = i + 1
                single_beds = x['single_beds'][i]
                double_beds = x['double_beds'][i]
                bunk_beds = x['bunk_beds'][i]
                tables = x['tables'][i]
                sofas = x['sofas'][i]
                model.constraint('single_beds[{room}] = {single_beds}'.format(**locals()))
                model.constraint('double_beds[{room}] = {double_beds}'.format(**locals()))
                model.constraint('bunk_beds[{room}] = {bunk_beds}'.format(**locals()))
                model.constraint('tables[{room}] = {tables}'.format(**locals()))
                model.constraint('sofas[{room}] = {sofas}'.format(**locals()))
        return model

    def infer(self, w, x=None, part=None, timeout=None, local=False, solver=None):
        """
        1) If local = True, perform inference only on the local subutility and
        not on the whole partial utility.
        2) We need to pass the timeout to the inference to make the comparison
        with CL
        """
        w_int = (w * 1000).astype(int)
        solver = self.solver if not solver else solver
        if part is None:
            if timeout is None:
                return pymzn.minizinc(_infer(PROBLEM + FULL), data={**self.args, 'w': w_int}, solver=solver, timeout=timeout, keep=self.keep, force_flatten=True)[0]
            else:
                solns = []
                while len(solns) == 0:
                    solns = pymzn.minizinc(_infer(PROBLEM + FULL), data={**self.args, 'w': w_int}, solver=solver, timeout=timeout, keep=self.keep, force_flatten=True)
                    timeout += 1
                return solns[0]
        if local:
            model = self._partial_model(_infer(PROBLEM), part, x)
            model = self._local_subutility(model, part)
        else:
            model = self._partial_model(_infer(PROBLEM + FULL), part, x)
        if timeout is None:
            return pymzn.minizinc(model, data={**self.args, 'w': w_int}, solver=solver, timeout=timeout, keep=self.keep, force_flatten=True)[0]
        else:
            solns = []
            while len(solns) == 0:
                solns = pymzn.minizinc(model, data={**self.args, 'w': w_int}, solver=solver, timeout=timeout, keep=self.keep, force_flatten=True)
                timeout += 1
            return solns[0]

    def improve(self, w, x, part=None, alpha=0.1, improve_margin=None, timeout=None, solver=None):
        w_int = (w * 1000).astype(int)
        solver = self.solver if not solver else solver
        timeout = 5
        if improve_margin is None:
            print('Improve: using alpha * regret')
            x_star = self.infer(w, x=x, part=part, timeout=timeout, solver=solver)
            regret = w_int.dot(self.phi(x_star) - self.phi(x))
            if regret == 0:
                return x
            margin = int(w_int.dot(self.phi(x)) + alpha * regret)
        else:
            print('Improve: using improve_margin')
            margin = int(w_int.dot(self.phi(x)) + 1000 * improve_margin)
        while True:
            try:
                if part is None:
                    if timeout is None:
                        return pymzn.minizinc(_improve(PROBLEM + FULL, margin),
                                            data={**self.args, 'w': w_int}, solver=solver, timeout=timeout, keep=self.keep, force_flatten=True)[0]
                    else:
                        solns = []
                        while len(solns) == 0:
                            solns = pymzn.minizinc(_improve(PROBLEM + FULL, margin),
                                                data={**self.args, 'w': w_int}, solver=solver, timeout=timeout, keep=self.keep, force_flatten=True)
                            timeout += 1
                        return solns[0]
                model = self._partial_model(_improve(PROBLEM + FULL, margin), part, x)
                if timeout is None:
                    return pymzn.minizinc(model, data={**self.args, 'w': w_int}, solver=solver, timeout=timeout, keep=self.keep, force_flatten=True)[0]
                else:
                    solns = []
                    while len(solns) == 0:
                        solns = pymzn.minizinc(model, data={**self.args, 'w': w_int}, solver=solver, timeout=timeout, keep=self.keep, force_flatten=True)
                        timeout += 1
                    return solns[0]
            except pymzn.MiniZincUnknownError:
                if timeout >= 20:
                    if improve_margin < 0.1:
                        return x
                    else:
                        improve_margin *= 0.5
                        print('Fixing improve margin: {}'.format(improve_margin))
                        margin = int(w_int.dot(self.phi(x)) + 1000 * improve_margin)
                else:
                    timeout *= 2
                    print('Fixing timeout: {}'.format(timeout))
            except pymzn.MiniZincUnsatisfiableError:
                print('Unsat, returning x')
                return x

