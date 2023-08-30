import re
import sys
from typing import Tuple

import networkx as nx
import random

import torch
from torch import DoubleTensor as Tensor
from torch import rand
from torch import nn

LOOP_COOP_DEFAULT = 1


def _equal(n1, n2) -> bool:
    """
    Test whether two structures have identical connectivity
    :param n1: nx.Graph
    :param n2: nx.Graph
    :return: Boolean indicating whether or not graphs are equal
    """
    nm = nx.algorithms.isomorphism.categorical_node_match("label", None)
    int_n1 = nx.convert_node_labels_to_integers(n1, label_attribute="label")
    int_n2 = nx.convert_node_labels_to_integers(n2, label_attribute="label")
    return nx.is_isomorphic(int_n1, int_n2, node_match=nm)


def gtostr(g: nx.DiGraph) -> str:
    """
    get string representation of sorted graph node set.
    :param g: input graph
    :return: string label of graph nodes
    """
    stout = ""
    for n in g.nodes():
        stout += str(n)
    # make invariant
    stout = ''.join(sorted(stout))
    return stout


class ReactionNetwork:
    """
    ReactionNetwork objects store all the information needed to run a simulation. It
    stores allowed reactions, intermediate structures, rate constants, and reaction free
    energy scores, all encoded in an attributed directed acyclic graph.
    The reaction network also stores optimization parameters explicitly in the parameters
    attribute.
    More then just being a container for reaction data, the ReactionNetwork class provides
    methods to build a network that prohibits stericly hindered interactions from a simple
    bngl file containing just pairwise interactions.

    Attributes:
        network: nx.DiGraph
            The networkx graph object that encodes allowed reactions in its structure.
            Nodes are structures (including all possible intermediates), and also
            store the copy number for the structure, and a graph layout of the structure.
            An edge indicates that one node may react to produce another, if more than one
            reactant is needed to to produce a product, then both edges will have the
            same uid attribute. Edges also store k_on and k_off.
            Nodes also store the rosetta score of that state, and edges store the delta
            score for that reaction. Note that all energy related attributes will be null
            until the reactionNetwork is processed by an energy explorer.

        allowed_edges: set[Tuple[int]]
            A set containing the allowed pairwise reactions defined in the input file.

        is_one_step: bool
            Whether to model loop closure as a one step reaction (potentially better for
            for larger, more "floppy", complexes) or two step (far less intermediate states,
            rosetta energies map more accurately)

    """
    def __init__(self, bngl_path: str, one_step: bool, seed=None):
        """
        :param bngl_path: path to bngl containing pairwise interactions.
        :param one_step: whether this reaction network should be built as one step or two step
        """
        self.network: nx.DiGraph() = nx.DiGraph()
        self.allowed_edges = {}
        self._node_count = 0
        self._rxn_count = 0
        self.num_monomers = 0
        self.is_one_step = one_step
        self.rxn_coupling = False
        self.uid_map = dict()
        self.boolCreation_rxn = False
        self.creation_species = []
        self.creation_nodes = []
        self.creation_rxn_data ={}
        self.titration_end_conc=-1
        self.default_k_creation = 1e-1
        self.boolDestruction_rxn = False
        self.destruction_species = []
        self.destruction_nodes = []
        self.destruction_rxn_data ={}
        self.default_k_destruction = 1e-1
        self.max_subunits = -1
        self.max_interactions = 2
        self.monomer_add_only = True
        self.chaperone=False
        self.homo_rates=False
        # default observables are monomers and final complex
        self.observables = dict()
        self.flux_vs_time = dict()
        self.seed = seed
        # resolve graph
        self._initial_copies = {}
        self.parse_bngl(open(bngl_path, 'r'), seed=self.seed)
        self.parameters = {}  # gradient params
        self.is_energy_set = True

        self.mon_rxns = dict()
        self.rxn_cid = dict()
        self.rxn_class = dict()     #Classifies reactions into number of bonds being formed. i.e. 1 bond - Dimer formation, 2-bonds - Trimer. Here dict format - no. of bonds:[list of uids]
        self.mon_rxn_map = dict()
        self.dG_map = dict()

    def get_params(self):
        """
        returns an iterator over optimization parameters
        :return:
        """
        for key in self.parameters:
            yield self.parameters[key]

    def get_reactant_sets(self, node_id: int):
        """
        Returns a iterator over coreactants for a given node (i.e. product)
        :param node_id: the node to know reactant sets for
        :return:
        """
        all_predecessors = set(self.network.in_edges(node_id))
        while len(all_predecessors) > 0:
            found = False
            reactant = all_predecessors.pop()
            predecessors = {reactant[0]}
            reactant_data = self.network[reactant[0]][reactant[1]]
            poss_coreactant = None
            # find complete reactant sets
            for poss_coreactant in all_predecessors:
                poss_coreactant_data = self.network[poss_coreactant[0]][poss_coreactant[1]]
                if reactant_data['uid'] == poss_coreactant_data['uid']:
                    found = True
                    break
            if found:
                all_predecessors.remove(poss_coreactant)
                predecessors.add(poss_coreactant[0])
            yield predecessors

    def parse_param(self, line):
        # Reserved Params
        # loop_coop: the loop cooperativity factor (f) = exp(-dG_coop / kb*T)
        #           range of 0 to 1, with binding being strongly forward at 1
        items = line.split(None, 1)
        items[1] = eval(items[1])
        print(items)
        if items[0] == 'default_assoc':
            self.default_k_on = items[1]
        elif items[0] == 'rxn_coupling':
            self.rxn_coupling = items[1]
            print(self.rxn_coupling)
        elif items[0] =='creation_rate':
            self.default_k_creation = items[1]
        elif items[0] =='destruction_rate':
            self.default_k_destruction = items[1]
        elif items[0] == 'max_subunits':
            self.max_subunits = items[1]
        elif items[0] == 'max_interactions':
            self.max_interactions = items[1]
        elif items[0] == 'monomer_add_only':
            self.monomer_add_only=items[1]
        elif items[0] == 'chaperone':
            self.chaperone=items[1]
            self.chaperone_rxns = []
            self.chap_uid_map = {}
            self.chap_int_spec_map = {}
            self.optimize_species={'substrate':[],'enz-subs':[]}
        elif items[0]== 'homo_rates':
            self.homo_rates=items[1]
        elif items[0]=='titration_time_int':
            print("Setting Titration End Point")
            self.titration_end_conc=items[1]
        return items

    def parse_species(self, line, params):
        items = line.split()
        sp_info = re.split('\\)|,|\\(', items[0])
        try:
            init_pop = int(items[1])
        except ValueError:
            try:
                init_pop = float(items[1])
            except ValueError:
                init_pop = int(params[items[1]])
        if self.max_subunits>0:
            print("Using multiGraph")
            state_net = nx.MultiGraph()
        else:
            state_net = nx.Graph()
        state_net.add_node(sp_info[0])
        print(state_net.nodes())
        print(init_pop)
        self.network.add_node(self._node_count, struct=state_net, copies=Tensor([float(init_pop)]),subunits=1)
        self._initial_copies[self._node_count] = Tensor([float(init_pop)])
        self._node_count += 1

    def parse_rule(self, line, params, seed=None, percent_negative=.5, score_range=100):
        items = re.split(r' |, ', line)
        print("Parsing rule...")
        #Old split
        # r_info = re.split('\\(.\\)+.|\\(.\\)<->', items[0])
        #New split
        #First splitting it by the reaction arrow. The second splitting the reactants side text to get species. Useful to identify creation ann destruction rxns.
        split_01 = re.split('<->',items[0])
        print("SPLIT_01: ",split_01)

        #Check if any of the reactants is in bound form
        if '!' in split_01[0]:
            r_info = re.split('\+',split_01[0])

            react_1 = "".join(re.split('\\(.\!.\\)|\.',r_info[0]))
            react_2 = "".join(re.split('|\\(.\\)|\+',r_info[1]))

            print(r_info)
        else:
            #If no bound reactants, then check if creation or destruction is present
            if 'null' in split_01:
                #Parsing for creation and Destruction done below
                pass
            else:
                #No bound reactnats, No creation or destruction. Parse reactants normally
                r_info = re.split('\\(.\\)+.|\\(.\\)',split_01[0])
                print(r_info)
                react_1 = r_info[0]
                react_2 = r_info[1]



        if params['default_assoc']:
            self.k_on = params['default_assoc']
        else:
            self.k_on = 1
        k_off = None
        if 'G=' in items[-1]:
            print("GGGGGGGGGgg")
            score = Tensor([float(items[-1].split('=')[1])])
        else:
            if seed:
                torch.random.manual_seed(seed)
            score = (rand(1, dtype=torch.double) - percent_negative) * score_range

        #After parsing reactions add to reaction network
        #Parsing for creation and destruction is done here
        if split_01[0]=='null':
            print("Found Creation rxn")
            species = re.split('\\(.\\)',split_01[1])[0]
            self.allowed_edges[tuple(['null',species])] = [None, None, LOOP_COOP_DEFAULT, score]
            self.boolCreation_rxn=True
            self.creation_species.append(species)
        elif split_01[1]=='null':
            print("Found Destruction rxn")
            species=re.split('\\(.\\)',split_01[0])[0]
            self.allowed_edges[tuple([species,'null'])] = [None, None, LOOP_COOP_DEFAULT, score]
            self.boolDestruction_rxn=True
            self.destruction_species.append(species)
        else:
            self.allowed_edges[tuple(sorted([react_1, react_2]))] = [None, None, LOOP_COOP_DEFAULT, score]


        if params.get('rxn_coupling') is not None:
            self.rxn_coupling=params['rxn_coupling']

    def parse_bngl(self, f, seed=None):
        """
        Read the bngl file and initialize allowed edges, and initialize the network with
        monomer nodes and copies.
        :param f: file object in read mode, pointed at input bngl
        :return: None
        """
        parameters = dict()
        cur_block = ''
        for line in f:
            line = line.strip()
            if len(line) > 0 and line[0] != '#':
                if "begin parameters" in line:
                    cur_block = 'param'
                elif "begin species" in line:
                    cur_block = 'species'
                elif "begin rules" in line:
                    cur_block = 'rules'
                elif "begin observables" in line:
                    cur_block = 'observables'
                elif "end" in line:
                    cur_block = ' '
                else:
                    if cur_block == 'param':
                        items = self.parse_param(line)
                        parameters[items[0]] = items[1]
                    elif cur_block == 'species':
                        self.parse_species(line, parameters)
                    elif cur_block == 'rules':
                        self.parse_rule(line, parameters, seed=None)


        # attach loop cooperativity param to rules (python 3.6+ only due to dict ordering changes)
        if "loop_coop" in parameters:
            if len(parameters['loop_coop']) != len(self.allowed_edges):
                raise ValueError('num loop_coop must equal to num allowed_edges')
            keys = list(self.allowed_edges.keys())
            for i, lcf in enumerate(parameters['loop_coop']):
                if lcf > 1 or lcf < 0:
                    raise ValueError('loop cooperativity factor must be between 0 and 1')
                self.allowed_edges[keys[i]][2] = lcf
        self.num_monomers = self._node_count

    def reset(self):
        """
        Initialize monomer copy numbers, and set all other species copy numbers to 0.
        :return:
        """
        for key in self._initial_copies:
            self.network.nodes[key]['copies'] = self._initial_copies[key]
        self.observables = {}
        self.flux_vs_time = {}
        # add default observables
        for i in range(self.num_monomers):
            self.observables[i] = (gtostr(self.network.nodes[i]['struct']), [])
            self.flux_vs_time[i] = (gtostr(self.network.nodes[i]['struct']), [])
        fin_dex = len(self.network.nodes) - 1
        self.observables[fin_dex] = (gtostr(self.network.nodes[fin_dex]['struct']), [])
        self.flux_vs_time[fin_dex] = (gtostr(self.network.nodes[fin_dex]['struct']), [])

    def intialize_activations(self, mode="middle"):
        """
        function to set and initialize k_on parameters for reaction network. Adds each to
        the parameters attribute, which stores all params to optimize over.
        :return:
        """
        if not self.is_energy_set:
            raise ValueError("The network free energies must be calculated for activation params to be used")
        for node in self.network.nodes:
            for reactant_set in self.get_reactant_sets(node):
                # same Tensor used in all three places (i.e. ptr)
                if mode == 'uniform':
                    k_on = nn.Parameter(rand(1, dtype=torch.double) * Tensor([1]), requires_grad=True)
                elif mode == 'middle':
                    k_on = nn.Parameter(Tensor([1.]), requires_grad=True)
                self.parameters[tuple(list(reactant_set) + [node])] = k_on
                for source in reactant_set:
                    self.network.edges[(source, node)]['k_on'] = k_on

    def initialize_random_pairwise_energy(self, percent_negative=.5, score_range=1000, seed=None):
        for node in self.network.nodes:
            for reactant_set in self.get_reactant_sets(node):
                if seed is not None:
                    torch.random.manual_seed(seed)
                score = (rand(1, dtype=torch.double) - percent_negative) * score_range
                for source in reactant_set:
                    if source < self.num_monomers:
                        self.network.edges[(source, node)]['rxn_score'] = score

        self.is_energy_set = True

    def _add_graph_state(self, connected_item: nx.Graph, source_1: int, source_2: int = None, template=None,subunits=1):
        """
        Adds a new species defined by connected_item to the graph, if unique.
        :param connected_item: The graph structure reoresenting the product (new node requested)
        :param source_1: reactant 1 node
        :param source_2: reactant 2 node (may be None)
        :param template_edge_id:
        :param subunits : No. of subunits in the new Node Complex. Added this parameters since it becomes easier to keep track of complex size in homo-oligomers
        :return:
        """
        if type(source_1) is not int:
            source_1 = int(source_1[0])
        if source_2 is not None and type(source_2) is not int:
            source_2 = int(source_2[0])
        node_exists = [x for x in self.network.nodes(data=True) if
                       _equal(x[1]['struct'], connected_item)]
        # print("Checking node exists for : ", connected_item.nodes())
        print(node_exists)
        print("Connected item Edges: ",connected_item.edges())
        new_edges_added = 0
        if len(node_exists) == 0:
            print("New node added--1")
            print(connected_item.nodes())
            self.network.add_node(self._node_count, struct=connected_item, copies=Tensor([0.]),subunits=subunits)
            self._initial_copies[self._node_count] = Tensor([0.])
            new_node = self._node_count
            self._node_count += 1
            # print(self.network.nodes())
        elif len(node_exists) > 1:
            raise Exception("Duplicate nodes in reaction Network")
        else:
            new_node = node_exists[0][0]
        if self.network.has_edge(source_1, new_node):
            # skip if edge exists failsafe.
            print("$$$$$$$")
            return None
        if not template:
            print("&&&&&&&")
            return None
        else:

            print("Adding an new edge--",source_1,new_node)
            # print("uid: ", self._rxn_count)

            #Creates a rxn_class, dG_map and monomer rxn map.
            #But it is based on number of bonds formed. Does not apply for different topologies
            #Instead this is now handles by the create_rxn_class

            # print("New bonds: ",template)
            # if len(template) in self.rxn_class.keys():
            #     self.rxn_class[len(template)].append(self._rxn_count)
            # else:
            #     self.rxn_class[len(template)] = [self._rxn_count]
            # if len(template) == 1:
            #     self.mon_rxn_map[template[0]]=self._rxn_count
            # else:
            #     rids = []
            #     for reactants in template:
            #         rids.append(self.mon_rxn_map[reactants])
            #     self.dG_map[self._rxn_count] = rids

            dg_coop = sum([self.allowed_edges[tuple(sorted(e))][3] for e in template])
            self.network.add_edge(source_1, new_node,
                                  k_on=self.default_k_on,
                                  k_off=None,
                                  lcf=1,
                                  rxn_score=dg_coop,
                                  uid=self._rxn_count)
            new_edges_added+=1
            if source_2 is not None:
                print("Adding an new edge--",source_2,new_node)
                # print("uid: ", self._rxn_count)
                self.network.add_edge(source_2, new_node,
                                      k_on=self.default_k_on,
                                      k_off=None,
                                      lcf=1,
                                      rxn_score=dg_coop,
                                      uid=self._rxn_count)
                new_edges_added+=1

                #Code for Coupling reactions
                #Here just creating a dictionary that contains monomer reactions mapped to unique ids (uid)
                #Check if both nodes are moomers
                if self.rxn_coupling and (self.network.nodes[source_1]['struct'].number_of_edges() == 0 and self.network.nodes[source_2]['struct'].number_of_edges() ==0):
                    #This will be true only when reactants are monomers. Since a monomer node is a graph with 0 edges.
                    #Checking the number of edges in the graph of each node is better since this discriminates between a homodimer which happens in the case we have repeated units.
                    #Now if they both are monomers, add it to the dictionary
                    reactants = tuple(sorted((source_1,source_2)))    #Cannot have list as keys in dictionary
                    self.mon_rxns[reactants]=self._rxn_count

                #Appending to uid map. This dictiobary maps each unique reaction id to it's reactants
                #{key - > rxn_count ; value - > (node1,node2)}
                self.uid_map[self._rxn_count] = tuple(sorted((source_1,source_2)))

            #Code for repeat units
            if len(template) > new_edges_added:
                print("The number of bonds formed are not compensated by the number of edges")
                print("This could be possible due to presence of a repeating subunit")

                #There could be a lot of controls to check this
                cmn_reactant = set(template[0])
                for b in range(len(template)-1):
                    cmn_reactant = cmn_reactant.intersection(template[b+1])

                if cmn_reactant:
                    cmn_reactant = cmn_reactant.pop()
                    print("The common reactant is: ",cmn_reactant)
                    cmn_node=-1
                    for node in self.network.nodes(data=True):
                        if cmn_reactant == gtostr(node[1]['struct']) and (node[1]['struct'].number_of_edges()==0):
                            #Get node number from node label.
                            #Make sure it is a monomer
                            cmn_node = node[0]
                    print("Edge added between: ", cmn_node,new_node)
                    self.network.add_edge(cmn_node, new_node,
                                  k_on=self.default_k_on,
                                  k_off=None,
                                  lcf=1,
                                  rxn_score=dg_coop,
                                  uid=self._rxn_count)
                    new_edges_added+=1

        self._rxn_count += 1
        if len(node_exists) == 0:
            print("New node added--2")
            print(self.network.nodes())
            return (new_node, self.network.nodes[new_node])
        else:
            return None

    def match_maker(self, n1, n2=None, one_step=False) -> list:
        """
        determines if a valid edge can be added between two network states, and preforms
        addition if possible
        :param one_step: whether to do one step binding
        :param n1: node in network
        :param n2: node in network
        :return:
        """
        nodes_added = []
        orig = n1[1]['struct']

        if n2 is not None:
            nextn = n2[1]['struct']
            item = nx.compose(orig, nextn)
            print("Orig edges: ",orig.edges())
            print("Nextn edges: ",nextn.edges())
            print("Item edges: ",item.edges())
        else:
            item = orig
        connected_item = item.copy()
        new_bonds = []
        add_to_graph = False
        complex_size = 0
        for poss_edge in list(self.allowed_edges.keys()):
            print("Allowed edges: ")
            print(poss_edge)
            # print(item.nodes())
            print([item.has_node(n) for n in poss_edge])
            print(item.has_edge(poss_edge[0], poss_edge[1]))
            if False not in [item.has_node(n) for n in poss_edge] and \
                    (n2 is None or
                     (True in [orig.has_node(n) for n in poss_edge] and
                      True in [nextn.has_node(n) for n in poss_edge]))\
                    and not item.has_edge(poss_edge[0], poss_edge[1]):
                print("############################3")
                repeat_units=False

                if self.monomer_add_only==True:
                    if (orig.number_of_edges() ==0 or nextn.number_of_edges() ==0):
                        # if n2 is None and True in [item.has_node(n) for n in poss_edge]: #For internal bonds, treated as repeating Units
                        #     #Add a node
                        #     print(connected_item.nodes())
                        #     connected_item = nx.relabel_nodes(connected_item,{poss_edge[0]:[poss_edge[0],poss_edge[1]]})
                        #     print(connected_item.nodes())
                        #     # connected_item.add_edge(poss_edge[0], poss_edge[0]+poss_edge[1])
                        #     # print(connected_item.nodes())
                        #     new_bonds.append(poss_edge)
                        #     add_to_graph=True
                        # else:

                        if self.chaperone and True in [item.has_node(sp) for sp in list(self.chap_int_spec_map.keys())]:
                            #If there are chaperone rxns, then there will a species present which has an intermediate which has the chaperone on it.
                            #Then this species cannot add more bonds to it. It should only dissociate.
                            continue
                        connected_item.add_edge(poss_edge[0], poss_edge[1])
                        new_bonds.append(poss_edge)
                        complex_size += n1[1]['subunits']
                        if n2 is not None:
                            complex_size+=n2[1]['subunits']


                        #Checking if the graph has a repeated units.
                        #If one node is AA and other is B, then B should form bonds with both A units.
                        #So another edge has to be added with same reactants.
                        #Can identify this by checking the edges and not nodes. Since "AA" node does not appear, but A-A eddge is allowed
                        for (u,v) in item.edges:
                            if u==v:
                                print("Repeat Units")
                                repeat_units = True
                        if repeat_units:
                            connected_item.add_edge(poss_edge[1], poss_edge[0])
                            new_bonds.append(poss_edge)

                        add_to_graph=True
                        # print("Connected Nodes: ",connected_item.nodes())
                        # print("Connected Edges: ",connected_item.edges())
                elif self.monomer_add_only==-1:
                    if self.chaperone and True in [item.has_node(sp) for sp in list(self.chap_int_spec_map.keys())]:
                        #If there are chaperone rxns, then there will a species present which has an intermediate which has the chaperone on it.
                        #Then this species cannot add more bonds to it. It should only dissociate.
                        continue
                    connected_item.add_edge(poss_edge[0], poss_edge[1])
                    new_bonds.append(poss_edge)
                    complex_size += n1[1]['subunits']
                    if n2 is not None:
                        complex_size+=n2[1]['subunits']


                    #Checking if the graph has a repeated units.
                    #If one node is AA and other is B, then B should form bonds with both A units.
                    #So another edge has to be added with same reactants.
                    #Can identify this by checking the edges and not nodes. Since "AA" node does not appear, but A-A eddge is allowed
                    for (u,v) in item.edges:
                        if u==v:
                            print("Repeat Units")
                            repeat_units = True
                    if repeat_units:
                        connected_item.add_edge(poss_edge[1], poss_edge[0])
                        new_bonds.append(poss_edge)

                    add_to_graph=True
                else:
                    if (orig.number_of_edges() > 0 and nextn.number_of_edges() >0):
                        if self.chaperone and True in [item.has_node(sp) for sp in list(self.chap_int_spec_map.keys())]:
                            #If there are chaperone rxns, then there will a species present which has an intermediate which has the chaperone on it.
                            #Then this species cannot add more bonds to it. It should only dissociate.
                            continue
                        connected_item.add_edge(poss_edge[0], poss_edge[1])
                        new_bonds.append(poss_edge)
                        complex_size += n1[1]['subunits']
                        if n2 is not None:
                            complex_size+=n2[1]['subunits']


                        #Checking if the graph has a repeated units.
                        #If one node is AA and other is B, then B should form bonds with both A units.
                        #So another edge has to be added with same reactants.
                        #Can identify this by checking the edges and not nodes. Since "AA" node does not appear, but A-A eddge is allowed
                        for (u,v) in item.edges:
                            if u==v:
                                print("Repeat Units")
                                repeat_units = True
                        if repeat_units:
                            connected_item.add_edge(poss_edge[1], poss_edge[0])
                            new_bonds.append(poss_edge)

                        add_to_graph=True


            elif True in [item.has_node(n) for n in poss_edge] and (n2 is None) and item.has_edge(poss_edge[0], poss_edge[1]):
                #Here you are checking that for this reaction, one edge already exists in the network. And if the edges exists
                #then so should all the reactants as nodes. Which is checked by the first conditional.
                #This condition is to necessary if there are repeating subunits in a complex. Because there are repeating subunits,
                #new bonds are not formed because between new incoming subunit and another existing subunit since the "if" statement checks
                #for existing edges. But now with this conditional we can form the extra bond.

                #TODO : This part of code needs to addressed in a diff part. This is a very messy way of doing. It should be done in Steric hindrance detected region.
                #Basically we are creating the reaction AB + A -> AAB. But this is done by trying to form internal bonds of AB. This should actually be done when two nodes AB
                #and A are evaluated in resolve_tree()

                print("*********************************************")
                print("Adding extra new bonds for the repeating unit - ")
                print(n1[1]['struct'].edges())
                print(add_to_graph)
                new_bonds.append(poss_edge)
                complex_size+=n1[1]['subunits']   #Only for n1 since n2 is None

            elif (n2 is not None) and (True in [orig.has_node(n) for n in poss_edge] and True in [nextn.has_node(n) for n in poss_edge]) and item.has_edge(poss_edge[0], poss_edge[1]):
                #TODO: Need to insert a condition which checks that this part of the code is only executed if there is a repeating unit in the complex.
                print("Item already has edge")
                #Once you see that item already has edge. Check if the complex has max. subunits
                #Find no. of edges in the node
                #This has to be done separately for monomers and non-monomers
                #It also depends on the no. of interactions one sub-unit can have.
                #A linear polymer model will be where each subunit has two interactions(2 interfaces)
                #A branched polymer will be when the subunit can have more than two interactions

                #CHeck if one node is monomer
                if orig.number_of_edges() ==0 or nextn.number_of_edges() ==0:
                    print("One of the reactants is a monomer")

                    n_edges = orig.number_of_edges() if orig.number_of_edges() else nextn.number_of_edges()
                    print(n_edges)
                    print(connected_item.edges())
                    total_subunits = n1[1]['subunits'] + n2[1]['subunits']


                    if total_subunits <= self.max_subunits:
                        print("There is room to add this subunit")

                        complex_size=total_subunits


                        #TO calculate the no. of new bonds to add, we have to loop over each reaction set and see if there exists a reaction between them:
                        #ALso we have to make sure it does not exceed max_interactions for the new subunits
                        e1 = orig.number_of_edges()
                        e2 = nextn.number_of_edges()
                        reactant_set = tuple([r1 for r1 in orig.nodes()] + [r2 for r2 in nextn.nodes()])    #A bit convoluted way of getting the node from each Graph. Have to individually loop over them and join.
                        print("Reactant Set: ",reactant_set)
                        if reactant_set == poss_edge:
                            #If its a linear polymer. Then only one new bond is formed. Chain elongation
                            #Since we don't know if n1 is monomer or n2, right now just looping over both subunits to add bonds.
                            if self.max_interactions ==2:
                                print("NEW BOND ADDEDDDDDDD")
                                new_bonds.append(poss_edge)
                                connected_item.add_edge(poss_edge[1], poss_edge[0])
                                if total_subunits == self.max_subunits:
                                    #Checking if addition of one more subunit leads to max-subunits.
                                    #This means it is a loop closure. So add one more bond
                                    print("LOOP CLOSUREEEEE")
                                    new_bonds.append(poss_edge)
                                    connected_item.add_edge(poss_edge[1], poss_edge[0])
                                print(connected_item.edges())
                            else:
                                print("Forming bonds to achieve max interactions from each sub-unit")
                                for i in range(n1[1]['subunits']):
                                    for j in range(n2[1]['subunits']):
                                        print("NEW BOND ADDEDDDDDDD")
                                        new_bonds.append(poss_edge)
                                        connected_item.add_edge(poss_edge[1], poss_edge[0])

                                # #In case if more interactions are allowed (max capacity), then we need to add more bonds
                                # if self.max_interactions>2:
                                #     #More bonds can be added
                                #     #How many more?
                                #     print("Can reach max occupancy")
                                #     max_bonds = math.comb(total_subunits,2)
                                #     for i in range(max_bonds-len(new_bonds)):
                                #         print("New Bond added..")
                                #         new_bonds.append(poss_edge)
                                # else:
                                #     #When max_interactions are 2 i.e. Linear polymer. Have to add one more bond for loop closure.
                                #     if total_subunits == self.max_subunits:
                                #         print("HULLLA BALLOOO!!")

                                    # new_bonds.append(poss_edge)
                        add_to_graph=True

                else:
                    #Both nodes are not monomers
                    print("BOTH NODES ARE NOT MONOMERS!!")
                    if self.monomer_add_only == -1:
                        print("NON-MONOMER ADDITION!!!!!")
                        #Only add reaction if user defined as adding rxn b/w non-monomers

                        #Before adding any edges, one modification has to be done in the new connected_item network graph
                        #Since the the connected_item is a union of orig and nextn nodes, if edges are common in nextn, then they are excluded.
                        #For e.g. A-A  + A-A  -> A-A-A-A ; Edges from only one AA complex will be included
                        #Irrespective of the final topology, the initial number of bonds should include one edge from each AA complex
                        #Loop over edges of each node and check if common
                        #Add the edge from nextn
                        for edge2 in nextn.edges():
                            connected_item.add_edge(edge2[0],edge2[1])

                        total_subunits = n1[1]['subunits'] + n2[1]['subunits']
                        if total_subunits <= self.max_subunits:
                            print("There is room to add this COMPLEX")
                            complex_size=total_subunits
                            reactant_set = tuple([r1 for r1 in orig.nodes()] + [r2 for r2 in nextn.nodes()])    #A bit convoluted way of getting the node from each Graph. Have to individually loop over them and join.
                            print("Reactant Set: ",reactant_set)
                            print(connected_item.edges())
                            if reactant_set == poss_edge:
                                #If its a linear polymer. Then only one new bond is formed. Chain elongation
                                #Since we don't know if n1 is monomer or n2, right now just looping over both subunits to add bonds.
                                if self.max_interactions ==2:
                                    print("NEW BOND ADDEDDDDDDD")
                                    new_bonds.append(poss_edge)
                                    connected_item.add_edge(poss_edge[1], poss_edge[0])
                                    if total_subunits == self.max_subunits:
                                        #Checking if addition of one more subunit leads to max-subunits.
                                        #This means it is a loop closure. So add one more bond
                                        print("LOOP CLOSUREEEEE")
                                        new_bonds.append(poss_edge)
                                        connected_item.add_edge(poss_edge[1], poss_edge[0])
                                    print(connected_item.edges())
                                else:
                                    print("Forming bonds to achieve max interactions from each sub-unit")
                                    for i in range(n1[1]['subunits']):
                                        for j in range(n2[1]['subunits']):
                                            print("NEW BOND ADDEDDDDDDD")
                                            new_bonds.append(poss_edge)
                                            connected_item.add_edge(poss_edge[1], poss_edge[0])

                            add_to_graph=True

            elif (True in [item.has_node(n) for n in poss_edge]) and (True in [len(n)>1 for n in poss_edge]) and self.chaperone and n2 is not None:


                #Chaperone
                #The previous conditionals where it is checking if a node exists using reactants from poss_edge does not work when one of the reactant in poss_edge is a complex
                #Because item represents the structure of a node. And it will always be composed of monomers as nodes.

                #You need to check this condition if the complex exists in the main network
                rxn_is_possible=False
                node_labels= (gtostr(orig),gtostr(nextn))

                if set(node_labels) == set(poss_edge):
                    print("*******Chaperone Reaction**********")
                    reactants = sorted((n1[0],n2[0]))
                    products = sorted(list(item.nodes()))
                    print(reactants,products)

                    if (reactants,products) not in self.chaperone_rxns:
                        #Adding reactants and products for the final step of chaperone rxn
                        self.chaperone_rxns.append((reactants,products))

                        #Add an enzyme subtrate complex to the graph state. This is the MM model

                        connected_item = item.copy()
                        new_bonds.append(poss_edge)

                        sp_len = [len(e) for e in poss_edge]     #Here poss_edge has two elements. One is the chaperone (len ==1) and one is the intermediate (len>1)
                        connected_item.add_edge(poss_edge[sp_len.index(1)][0],poss_edge[sp_len.index(1)])     #Connecting internal T-T edge. Need to check this. Does not affect any opt since the edges in the network are added properly.

                        if poss_edge[sp_len.index(1)] not in self.chap_int_spec_map:
                            self.chap_int_spec_map[poss_edge[sp_len.index(1)]] =  [self._node_count]
                        else:
                            self.chap_int_spec_map[poss_edge[sp_len.index(1)]].append(self._node_count)

                        add_to_graph=True

                continue
        # resolving one step  network
        if one_step and add_to_graph:
            new_node = self._add_graph_state(connected_item, n1, source_2=n2, template=new_bonds,subunits=complex_size)
            if new_node is not None:
                nodes_added.append(new_node)

        return nodes_added

    def is_hindered(self, n1, n2) -> bool:
        """
        Determines if binding two species would be sterically hindered.
        :param n1: node 1 (species 1)
        :param n2: node 2 (species 2)
        :return:
        """
        node_set1 = set(n1[1]['struct'].nodes())
        node_set2 = set(n2[1]['struct'].nodes())
        print("-----")
        print(node_set1)
        print(node_set2)
        print(node_set1 - node_set2)
        return len(node_set1 - node_set2) < len(node_set1)

    def decompose_monomers(self,n1,monomer_set):
        if len(self.network.in_edges(n1)) == 0:
            return(True,monomer_set)
        else:
            for incoming_edge in self.network.in_edges(n1):
                flag,monomer_set = self.decompose_monomers(incoming_edge[0],monomer_set)
                if flag:
                    monomer_set.append(incoming_edge[0])
            return(False,monomer_set)

    def map_coupled_rxns(self):
        cid={}
        for uid,reactants in self.uid_map.items():
            if self.network.nodes[reactants[0]]['struct'].number_of_edges() ==0 and self.network.nodes[reactants[1]]['struct'].number_of_edges() ==0:
                #Both are monomers. No coupling. Skip
                continue
            elif self.network.nodes[reactants[0]]['struct'].number_of_edges() ==0 :
                #Reactant 1 is monomer. Reactant 2 is not
                #Get all nodes of all monomer species
                flag,monomer_set = self.decompose_monomers(reactants[1],[])
                monomer_set = list(set(monomer_set))
                for mon in monomer_set:
                    rxn_pair = tuple(sorted((reactants[0],mon)))
                    if self.mon_rxns.get(rxn_pair) is not None:
                        mon_rxn_id = self.mon_rxns[rxn_pair]
                        if uid in cid.keys() :
                            if mon_rxn_id not in cid[uid]:
                                cid[uid].append(mon_rxn_id)
                        else:
                            cid[uid] = [mon_rxn_id]
            elif self.network.nodes[reactants[1]]['struct'].number_of_edges() ==0 :
                #Reactant 2 is monomer. Reactant 1 is not
                #Get all nodes of all monomer species
                flag,monomer_set = self.decompose_monomers(reactants[0],[])
                monomer_set = list(set(monomer_set))
                for mon in monomer_set:
                    rxn_pair = tuple(sorted((reactants[1],mon)))
                    if self.mon_rxns.get(rxn_pair) is not None:
                        mon_rxn_id = self.mon_rxns[rxn_pair]
                        if uid in cid.keys() :
                            if mon_rxn_id not in cid[uid]:
                                cid[uid].append(mon_rxn_id)
                        else:
                            cid[uid] = [mon_rxn_id]
            else:
                #Both reactants are not monomers
                #Get nodes of all monomer species for each complex
                flag1,monomer_set1 = self.decompose_monomers(reactants[0],[])
                flag2,monomer_set2 = self.decompose_monomers(reactants[1],[])
                monomer_set1 = list(set(monomer_set1))
                monomer_set2 = list(set(monomer_set2))

                for m1 in monomer_set1:
                    for m2 in monomer_set2:
                        rxn_pair = tuple(sorted((m1,m2)))
                        if self.mon_rxns.get(rxn_pair) is not None:
                            mon_rxn_id = self.mon_rxns[rxn_pair]
                            if uid in cid.keys():
                                if mon_rxn_id not in cid[uid]:
                                    cid[uid].append(mon_rxn_id)
                            else:
                                cid[uid]=[mon_rxn_id]
        return(cid)

    def check_if_node_exists(self,species):
        for node in self.network.nodes():
            node_label = gtostr(self.network.nodes[node]['struct'])
            if node_label == species:
                return(True)
        return(False)


    def resolve_creation_rxn(self):

        print(self.creation_species)
        for n in self.network.nodes:
            node_lb = gtostr(self.network.nodes()[n]['struct'])
            if (node_lb in self.creation_species) and self.network.nodes[n]['struct'].number_of_edges() ==0:    #The second condition is just to check if its a monomer and not a homodimer
                self.creation_nodes.append(n)
                self.creation_rxn_data[n] = {'uid':self._rxn_count,'k_on':self.default_k_creation}
                self._rxn_count+=1
            if (node_lb in self.destruction_species) and self.network.nodes[n]['struct'].number_of_edges() ==0:
                self.destruction_nodes.append(n)
                self.destruction_rxn_data[n] = {'uid':self._rxn_count,'k_on':self.default_k_destruction}
                self._rxn_count+=1

    def resolve_chaperone_rxn(self):
        print("Resolving Chaperone Rxns::")
        print(self.chaperone_rxns)
        for chap in self.chaperone_rxns:
            reactant = chap[0]    #Which nodes are reacting. e.g. AB + X
            products=[]
            enz_sub_complx = "".join(chap[1])   #Name of enzymen subtrate complex = ABX
            chap_species = -1
            for n in self.network.nodes():
                sp_label = gtostr(self.network.nodes[n]['struct'])
                if ( sp_label in chap[1]):
                    products.append(n)
                if (n in reactant) and (sp_label in list(self.chap_int_spec_map.keys())):
                    chap_species = n
                    for int_species in self.chap_int_spec_map[sp_label]:
                        if gtostr(self.network.nodes[int_species]['struct']) == enz_sub_complx:
                            r=int_species

                if (n in reactant) and len(sp_label)>1:
                    self.optimize_species['substrate'].append(n)



            print("Products:",products)
            print("Reactants: ",r)
            self.optimize_species['enz-subs'].append(r)

            for p in products:
                self.network.add_edge(r, p,
                              k_on=self.default_k_on,
                              k_off=None,
                              lcf=1,
                              rxn_score=torch.Tensor([float(-100)]),
                              uid=self._rxn_count)

            self.uid_map[self._rxn_count] = reactant
            if chap_species not in self.chap_uid_map:
                self.chap_uid_map[chap_species] = [self._rxn_count]
            else:
                self.chap_uid_map[chap_species].append(self._rxn_count)
            self._rxn_count+=1

            for edge in self.network.in_edges(r):
                data = self.network.get_edge_data(edge[0],edge[1])
                uid = data['uid']
                if uid not in self.chap_uid_map[chap_species]:
                    self.chap_uid_map[chap_species].append(uid)

    def create_rxn_class(self):
        uid_dict = {}
        uid_reactants = {}
        for n in self.network.nodes():
            #print(n)
            #print(rn.network.nodes()[n])
            for k,v in self.network[n].items():
                uid = v['uid']
                r1 = set(gtostr(self.network.nodes[n]['struct']))
                p = set(gtostr(self.network.nodes[k]['struct']))
                r2 = p-r1
                reactants = (r1,r2)
                uid_val = {'uid':uid,'reactants':reactants,'kon':v['k_on'],'score':v['rxn_score'],'koff':v['k_off']}
                uid_reactants[uid]=reactants
                if uid not in uid_dict.keys():
                    uid_dict[uid] = uid_val

        final_rxn_class = {}
        for key,rnts in sorted(uid_reactants.items()):
        #     print(key,"\t\t",rnts)

            l1 = len(rnts[0])
            l2 = len(rnts[1])


            if (l1,l2) in final_rxn_class.keys():
                final_rxn_class[(l1,l2)].append(key)
            elif (l2,l1) in final_rxn_class.keys():
                final_rxn_class[(l2,l1)].append(key)
            else:
                final_rxn_class[(l1,l2)] = [key]
        self.rxn_class = final_rxn_class


    def resolve_tree(self):
        """
        Build the full reaction network from whatever initial info was given
        :param is_one_step:
        :return:
        """
        new_nodes = list(self.network.nodes(data=True))
        while len(new_nodes) > 0:
            node = new_nodes.pop(0)
            for anode in list(self.network.nodes(data=True)):
                print("Node-1 : ",node)
                print("Node-2 : ",anode)
                if not self.is_hindered(node, anode):
                    print("False")
                    new_nodes += self.match_maker(node, anode, self.is_one_step)
                else:
                    print("Steric hindrance detected")
                    #Now it means there is some steric hindrance (which is decided by the fact that a new subunit to be added is already present in the complex.
                    #This is where a complex with multiple repeating sub units have to be resolved or homo-oligomers.
                    #To control this addition, we have to deine the max repeating units in a complex. If not there is a chance this will keep on expanding the complex

                    #Check condition if the node already has max subunits. This can be checked by the number of edges within the Graph of each node.
                    # if (node[1]['struct'].number_of_edges() < self.max_subunits - 1) and (anode[1]['struct'].number_of_edges() < self.max_subunits - 1) and self.max_subunits >0 :
                    #     print("Adding another subunit")
                    #     new_nodes+= self.match_maker(node,anode,self.is_one_step)
                    # elif ((node[1]['struct'].number_of_edges() >= self.max_subunits -1) or (anode[1]['struct'].number_of_edges() >= self.max_subunits-1)) and self.max_subunits >0:
                    #     print("Max subunits limit reached")
                    #     print(node[1]['struct'].edges())
                    #     print(anode[1]['struct'].edges())
                    if (node[1]['subunits']+anode[1]['subunits'] <= self.max_subunits) and (self.max_subunits >0):
                        print("Adding another subunit")
                        new_nodes+= self.match_maker(node,anode,self.is_one_step)
                    elif (node[1]['subunits']+anode[1]['subunits'] > self.max_subunits) and (self.max_subunits >0):
                        print("Max subunits limit reached")
                        print(node[1]['struct'].edges())
                        print(anode[1]['struct'].edges())


            # must also try internal bonds
            print("Trying internal bonds")
            new_nodes += self.match_maker(node,one_step=self.is_one_step)

        #Calculating dG of final complex
        #Add code here

        # add default observables
        #Add all nodes as observables
        for i in range(len(self.network.nodes)):
            self.observables[i] = (gtostr(self.network.nodes[i]['struct']), [])
            self.flux_vs_time[i] = (gtostr(self.network.nodes[i]['struct']), [])
        # fin_dex = len(self.network.nodes) - 1
        # self.observables[fin_dex] = (gtostr(self.network.nodes[fin_dex]['struct']), [])

        #Create rxn class dict; Used for parametrizing homogeneous model
        self.create_rxn_class()

        if self.rxn_coupling:
            self.rxn_cid = self.map_coupled_rxns()
            print("Coupling Reaction ID: ", self.rxn_cid)
        if self.boolCreation_rxn or self.boolDestruction_rxn:
            print("Resolving Creation and Destruction rxns")
            self.resolve_creation_rxn()
            print("Creation Reactions: ")
            print(self.creation_nodes)
            print(self.creation_rxn_data)
            print("Destructions Reactions: ")
            print(self.destruction_nodes)
            print(self.destruction_rxn_data)

        if self.chaperone:
            self.resolve_chaperone_rxn()

if __name__ == '__main__':
    bngls_path = sys.argv[1]  # path to bngl
    dt = float(sys.argv[2])  # time step in seconds
    iter = int(sys.argv[3])  # number of time steps to simulate
    m = ReactionNetwork(sys.argv[1],True)
    print('done')
