from KineticAssembly_AD import ReactionNetwork
from KineticAssembly_AD.reaction_network import gtostr
import numpy as np
from typing import Tuple, Union

import matplotlib
from matplotlib import pyplot as plt

from torch import DoubleTensor as Tensor

import sympy as sympy


def find_eq_eqns(rn):
    # ensure same elements reference the same symbols.
    sym_buf = {}
    constraints = {}
    for i in rn.network.nodes:
        name = frozenset(rn.network.nodes[i]['struct'].nodes())
        n = sympy.Symbol(gtostr(rn.network.nodes[i]['struct']))
        sym_buf[i] = n
        copies = rn.network.nodes[i]['copies']
        if copies > 0:
            constraints[name] = -1 * copies
        for key in constraints.keys():
            # if key is contained in this node
            if len(key.intersection(name)) == len(key):
                constraints[key] = n + constraints[key]
    eqn_list = list(constraints.values())
    for n in rn.network.nodes:
        c = sym_buf[n]
        for r_set in rn.get_reactant_sets(n):
            r_tup = tuple(r_set)
            data = rn.network.get_edge_data(r_tup[0], n)
            a = sym_buf[r_tup[0]]
            b = sym_buf[r_tup[1]]

            kon = data['k_on']
            koff = data['k_off']
            print("Off rates: ",koff)
            eqn = - a * b * kon + c * koff
            eqn_list.append(eqn)
    return eqn_list, sym_buf


class EquilibriumSolver:

    def __init__(self, net: ReactionNetwork):
        self.rn = net
        self.poly_system, self.symbols = find_eq_eqns(net)

    def solve(self, depth=0, init_val=[],verifyBool=True):
        #print(self.poly_system)
        if depth > 100:
            #raise ValueError('unable to find acceptable solution')
            print("No aceeptable solution found")
            return None
        copies = list(self.rn._initial_copies.values())
        if not init_val:
            init_val = (np.random.rand((len(copies))) * max(copies).clone().detach().numpy()).tolist()
        solution = None
        try:
            solution = sympy.solvers.nsolve(self.poly_system,
                                            list(self.symbols.values()),
                                            init_val,
                                            prec=7,
                                            max_steps=1000000000,
                                            verify=verifyBool)
        except ValueError:
            self.solve(depth=depth+1)

        return solution
