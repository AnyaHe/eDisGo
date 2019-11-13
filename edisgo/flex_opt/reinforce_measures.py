import copy
import math
import networkx as nx
from networkx.algorithms.shortest_paths.weighted import _dijkstra as \
    dijkstra_shortest_path_length
import pandas as pd
import numpy as np

from edisgo.network.components import Generator, Load
from edisgo.network.grids import LVGrid

import logging
logger = logging.getLogger('edisgo')


def extend_distribution_substation_overloading(edisgo_obj, critical_stations):
    """
    Reinforce MV/LV substations due to overloading issues.

    In a first step a parallel transformer of the same kind is installed.
    If this is not sufficient as many standard transformers as needed are
    installed.

    Parameters
    ----------
    edisgo_obj : :class:`~.edisgo.EDisGo`
    critical_stations : :pandas:`pandas.DataFrame<dataframe>`
        Dataframe containing over-loaded MV/LV stations, their apparent power
        at maximal over-loading and the corresponding time step.
        Index of the dataframe are the over-loaded stations of type
        :class:`~.network.components.LVStation`. Columns are 's_pfa'
        containing the apparent power at maximal over-loading as float and
        'time_index' containing the corresponding time step the over-loading
        occured in as :pandas:`pandas.Timestamp<timestamp>`. See
        :func:`~.flex_opt.check_tech_constraints.mv_lv_station_load` for more
        information.

    Returns
    -------
    dict
        Dictionary with lists of added and removed transformers.

    """

    # get parameters for standard transformer
    try:
        standard_transformer = edisgo_obj.equipment_data['lv_trafos'].loc[
            edisgo_obj.config['grid_expansion_standard_equipment'][
                'mv_lv_transformer']]
    except KeyError:
        print('Standard MV/LV transformer is not in equipment list.')

    transformers_changes = {'added': {}, 'removed': {}}
    for grid_name in critical_stations.index:
        grid = edisgo_obj.topology._grids[grid_name]
        # list of maximum power of each transformer in the station
        s_max_per_trafo = grid.transformers_df.s_nom

        # maximum station load from power flow analysis
        s_station_pfa = critical_stations.s_pfa[grid_name]

        # determine missing transformer power to solve overloading issue
        case = edisgo_obj.timeseries.timesteps_load_feedin_case[
            critical_stations.time_index[grid_name]]
        load_factor = edisgo_obj.config['grid_expansion_load_factors'][
            'lv_{}_transformer'.format(case)]
        s_trafo_missing = s_station_pfa/load_factor - sum(s_max_per_trafo)

        # check if second transformer of the same kind is sufficient
        # if true install second transformer, otherwise install as many
        # standard transformers as needed
        if max(s_max_per_trafo) >= s_trafo_missing:
            # if station has more than one transformer install a new
            # transformer of the same kind as the transformer that best
            # meets the missing power demand
            duplicated_transformer = grid.transformers_df.loc[
                grid.transformers_df[s_max_per_trafo >= s_trafo_missing][
                    's_nom'].idxmin()]
            name = duplicated_transformer.name.split('_')
            name.insert(-1, 'reinforced')
            name[-1] = len(grid.transformers_df) + 1
            duplicated_transformer.name = '_'.join([str(_) for _ in name])
            edisgo_obj.topology.transformers_df = \
                edisgo_obj.topology.transformers_df.append(
                    duplicated_transformer)

            transformers_changes['added'][grid_name] = \
                [duplicated_transformer.name]

        else:
            # get any transformer to get attributes for new transformer from
            duplicated_transformer = grid.transformers_df.iloc[0]
            name = duplicated_transformer.name.split('_')
            name.insert(-1, 'reinforced')
            duplicated_transformer.s_nom = standard_transformer.S_nom
            duplicated_transformer.r_pu = standard_transformer.r_pu
            duplicated_transformer.x_pu = standard_transformer.x_pu
            duplicated_transformer.type_info = standard_transformer.name
            # calculate how many parallel standard transformers are needed
            number_transformers = math.ceil(
                s_station_pfa / (standard_transformer.S_nom * load_factor))

            new_transformers = pd.DataFrame()
            # add transformer to station
            for i in range(number_transformers):
                name[-1] = i + 1
                duplicated_transformer.name = '_'.join([str(_) for _ in name])
                new_transformers = new_transformers.append(
                    duplicated_transformer)
            transformers_changes['added'][grid_name] = \
                new_transformers.index.values
            transformers_changes['removed'][grid_name] = \
                grid.transformers_df.index.values
            edisgo_obj.topology.transformers_df.drop(
                grid.transformers_df.index.values, inplace=True)
            edisgo_obj.topology.transformers_df = \
                edisgo_obj.topology.transformers_df.append(new_transformers)
    return transformers_changes


def extend_distribution_substation_overvoltage(network, critical_stations):
    """
    Reinforce MV/LV substations due to voltage issues.

    A parallel standard transformer is installed.

    Parameters
    ----------
    network : :class:`~.network.topology.Topology`
    critical_stations : :obj:`dict`
        Dictionary with :class:`~.network.grids.LVGrid` as key and a
        :pandas:`pandas.DataFrame<dataframe>` with its critical station and
        maximum voltage deviation as value.
        Index of the dataframe is the :class:`~.network.components.LVStation`
        with over-voltage issues. Columns are 'v_mag_pu' containing the
        maximum voltage deviation as float and 'time_index' containing the
        corresponding time step the over-voltage occured in as
        :pandas:`pandas.Timestamp<timestamp>`.

    Returns
    -------
    Dictionary with lists of added transformers.

    """

    # get parameters for standard transformer
    try:
        standard_transformer = network.equipment_data['lv_trafos'].loc[
            network.config['grid_expansion_standard_equipment'][
                'mv_lv_transformer']]
    except KeyError:
        print('Standard MV/LV transformer is not in equipment list.')

    transformers_changes = {'added': {}}
    for grid in critical_stations.keys():

        # get any transformer to get attributes for new transformer from
        station_transformer = grid.station.transformers[0]

        new_transformer = Transformer(
            id='LVStation_{}_transformer_{}'.format(
                str(grid.station.id), str(len(grid.station.transformers) + 1)),
            geom=station_transformer.geom,
            mv_grid=station_transformer.mv_grid,
            grid=station_transformer.grid,
            voltage_op=station_transformer.voltage_op,
            type=copy.deepcopy(standard_transformer))

        # add standard transformer to station and return value
        grid.station.add_transformer(new_transformer)
        transformers_changes['added'][grid.station] = [new_transformer]

    if transformers_changes['added']:
        logger.debug("==> {} LV station(s) has/have been reinforced ".format(
            str(len(transformers_changes['added']))) +
                    "due to overloading issues.")

    return transformers_changes


def extend_substation_overloading(edisgo_obj, critical_stations):
    """
    Reinforce HV/MV station due to overloading issues.

    In a first step a parallel transformer of the same kind is installed.
    If this is not sufficient as many standard transformers as needed are
    installed.

    Parameters
    ----------
    edisgo_obj : :class:`~.edisgo.EDisGo`
    critical_stations : pandas:`pandas.DataFrame<dataframe>`
        Dataframe containing over-loaded HV/MV stations, their apparent power
        at maximal over-loading and the corresponding time step.
        Index of the dataframe are the over-loaded stations of type
        :class:`~.network.components.MVStation`. Columns are 's_pfa'
        containing the apparent power at maximal over-loading as float and
        'time_index' containing the corresponding time step the over-loading
        occured in as :pandas:`pandas.Timestamp<timestamp>`. See
        :func:`~.flex_opt.check_tech_constraints.hv_mv_station_load` for more
        information.

    Returns
    -------
    Dictionary with lists of added and removed transformers.

    """
    if len(critical_stations) > 1:
        raise Exception(
            "More than one MV station to extend was given. "
            "There should only exist one station, please check.")

    # get parameters for standard transformer
    try:
        standard_transformer = edisgo_obj.equipment_data['mv_trafos'].loc[
            edisgo_obj.config['grid_expansion_standard_equipment'][
                'hv_mv_transformer']]
    except KeyError:
        print('Standard HV/MV transformer is not in equipment list.')

    transformers_changes = {'added': {}, 'removed': {}}
    # list of maximum power of each transformer in the station
    trafos = edisgo_obj.topology.transformers_hvmv_df
    s_max_per_trafo = trafos.s_nom

    # maximum station load from power flow analysis
    s_station_pfa = critical_stations.s_pfa[0]

    # determine missing transformer power to solve overloading issue
    case = edisgo_obj.timeseries.timesteps_load_feedin_case[
        critical_stations.time_index[0]]
    load_factor = edisgo_obj.config['grid_expansion_load_factors'][
        'mv_{}_transformer'.format(case)]
    s_trafo_missing = s_station_pfa/load_factor - sum(s_max_per_trafo)

    # check if second transformer of the same kind is sufficient
    # if true install second transformer, otherwise install as many
    # standard transformers as needed
    if max(s_max_per_trafo) >= s_trafo_missing:
        # if station has more than one transformer install a new
        # transformer of the same kind as the transformer that best
        # meets the missing power demand
        duplicated_transformer = trafos.loc[
            trafos[s_max_per_trafo > s_trafo_missing]['s_nom'].idxmin()]
        name = duplicated_transformer.name.split('_')
        name.insert(-1, 'reinforced')
        name[-1] = len(trafos) + 1
        duplicated_transformer.name = '_'.join([str(_) for _ in name])
        edisgo_obj.topology.transformers_hvmv_df = \
            edisgo_obj.topology.transformers_hvmv_df.append(
                duplicated_transformer)

        transformers_changes['added'][critical_stations.index[0]] = \
            [duplicated_transformer.name]

    else:
        # get any transformer to get attributes for new transformer from
        duplicated_transformer = trafos.iloc[0]
        name = duplicated_transformer.name.split('_')
        name.insert(-1, 'reinforced')
        duplicated_transformer.s_nom = standard_transformer.S_nom
        duplicated_transformer.type_info = standard_transformer.name
        # calculate how many parallel standard transformers are needed
        number_transformers = math.ceil(
            s_station_pfa / (standard_transformer.S_nom * load_factor))

        new_transformers = pd.DataFrame()
        # add transformer to station
        for i in range(number_transformers):
            name[-1] = i+1
            duplicated_transformer.name = '_'.join([str(_) for _ in name])
            new_transformers = new_transformers.append(duplicated_transformer)
        new_transformers.set_index('name')
        transformers_changes['added'][
            critical_stations.index[0]] = new_transformers.index.values
        transformers_changes['removed'][
            critical_stations.index[0]] = trafos.index.values
        edisgo_obj.topology.transformers_hvmv_df = new_transformers

    if transformers_changes['added']:
        logger.debug("==> MV station has been reinforced due to overloading "
                     "issues.")

    return transformers_changes


def reinforce_branches_overvoltage(network, grid, crit_nodes):
    """
    Reinforce MV and LV topology due to voltage issues.

    Parameters
    ----------
    network : :class:`~.network.network.Network`
    grid : :class:`~.network.grids.MVGrid` or :class:`~.network.grids.LVGrid`
    crit_nodes : :pandas:`pandas.DataFrame<dataframe>`
        Dataframe with critical nodes, sorted descending by voltage deviation.
        Index of the dataframe are nodes (of type
        :class:`~.network.components.Generator`, :class:`~.network.components.Load`,
        etc.) with over-voltage issues. Columns are 'v_mag_pu' containing the
        maximum voltage deviation as float and 'time_index' containing the
        corresponding time step the over-voltage occured in as
        :pandas:`pandas.Timestamp<timestamp>`.

    Returns
    -------
    Dictionary with :class:`~.network.components.Line` and the number of lines
    added.

    Notes
    -----
    Reinforce measures:

    1. Disconnect line at 2/3 of the length between station and critical node
    farthest away from the station and install new standard line
    2. Install parallel standard line

    In LV grids only lines outside buildings are reinforced; loads and
    generators in buildings cannot be directly connected to the MV/LV station.

    In MV grids lines can only be disconnected at LVStations because they
    have switch disconnectors needed to operate the lines as half rings (loads
    in MV would be suitable as well because they have a switch bay (Schaltfeld)
    but loads in dingo are only connected to MV busbar). If there is no
    suitable LV station the generator is directly connected to the MV busbar.
    There is no need for a switch disconnector in that case because generators
    don't need to be n-1 safe.

    """

    # load standard line data
    if isinstance(grid, LVGrid):
        try:
            standard_line = network.equipment_data['lv_cables'].loc[
                network.config['grid_expansion_standard_equipment']['lv_line']]
        except KeyError:
            print('Chosen standard LV line is not in equipment list.')
    else:
        try:
            standard_line = network.equipment_data['mv_cables'].loc[
                network.config['grid_expansion_standard_equipment']['mv_line']]
        except KeyError:
            print('Chosen standard MV line is not in equipment list.')

    # find first nodes of every main line as representatives
    rep_main_line = list(
        nx.predecessor(grid.graph, grid.station, cutoff=1).keys())
    # list containing all representatives of main lines that have already been
    # reinforced
    main_line_reinforced = []

    lines_changes = {}
    for node in crit_nodes.index:
        path = nx.shortest_path(grid.graph, grid.station, node)
        # raise exception if voltage issue occurs at station's secondary side
        # because voltage issues should have been solved during extension of
        # distribution substations due to overvoltage issues.
        if len(path) == 1:
            logging.error("Voltage issues at busbar in LV network {} should have "
                          "been solved in previous steps.".format(grid))
        else:
            # check if representative of line is already in list
            # main_line_reinforced; if it is, the main line the critical node
            # is connected to has already been reinforced in this iteration
            # step
            if not path[1] in main_line_reinforced:

                main_line_reinforced.append(path[1])
                # get path length from station to critical node
                get_weight = lambda u, v, data: data['line'].length
                path_length = dijkstra_shortest_path_length(
                    grid.graph, grid.station, get_weight, target=node)
                # find first node in path that exceeds 2/3 of the line length
                # from station to critical node farthest away from the station
                node_2_3 = next(j for j in path if
                                path_length[j] >= path_length[node] * 2 / 3)

                # if LVGrid: check if node_2_3 is outside of a house
                # and if not find next BranchTee outside the house
                if isinstance(grid, LVGrid):
                    if isinstance(node_2_3, BranchTee):
                        if node_2_3.in_building:
                            # ToDo more generic (new function)
                            try:
                                node_2_3 = path[path.index(node_2_3) - 1]
                            except IndexError:
                                print('BranchTee outside of building is not ' +
                                      'in path.')
                    elif (isinstance(node_2_3, Generator) or
                              isinstance(node_2_3, Load)):
                        pred_node = path[path.index(node_2_3) - 1]
                        if isinstance(pred_node, BranchTee):
                            if pred_node.in_building:
                                # ToDo more generic (new function)
                                try:
                                    node_2_3 = path[path.index(node_2_3) - 2]
                                except IndexError:
                                    print('BranchTee outside of building is ' +
                                          'not in path.')
                    else:
                        logging.error("Not implemented for {}.".format(
                            str(type(node_2_3))))
                # if MVGrid: check if node_2_3 is LV station and if not find
                # next LV station
                else:
                    if not isinstance(node_2_3, LVStation):
                        next_index = path.index(node_2_3) + 1
                        try:
                            # try to find LVStation behind node_2_3
                            while not isinstance(node_2_3, LVStation):
                                node_2_3 = path[next_index]
                                next_index += 1
                        except IndexError:
                            # if no LVStation between node_2_3 and node with
                            # voltage problem, connect node directly to
                            # MVStation
                            node_2_3 = node

                # if node_2_3 is a representative (meaning it is already
                # directly connected to the station), line cannot be
                # disconnected and must therefore be reinforced
                if node_2_3 in rep_main_line:
                    crit_line = grid.graph.get_edge_data(
                        grid.station, node_2_3)['line']

                    # if critical line is already a standard line install one
                    # more parallel line
                    if crit_line.type.name == standard_line.name:
                        crit_line.quantity += 1
                        lines_changes[crit_line] = 1

                    # if critical line is not yet a standard line replace old
                    # line by a standard line
                    else:
                        # number of parallel standard lines could be calculated
                        # following [2] p.103; for now number of parallel
                        # standard lines is iterated
                        crit_line.type = standard_line.copy()
                        crit_line.quantity = 1
                        crit_line.kind = 'cable'
                        lines_changes[crit_line] = 1

                # if node_2_3 is not a representative, disconnect line
                else:
                    # get line between node_2_3 and predecessor node (that is
                    # closer to the station)
                    pred_node = path[path.index(node_2_3) - 1]
                    crit_line = grid.graph.get_edge_data(
                        node_2_3, pred_node)['line']
                    # add new edge between node_2_3 and station
                    new_line_data = {'line': crit_line,
                                     'type': 'line'}
                    grid.graph.add_edge(grid.station,node_2_3, **new_line_data)
                    # remove old edge
                    grid.graph.remove_edge(pred_node, node_2_3)
                    # change line length and type
                    crit_line.length = path_length[node_2_3]
                    crit_line.type = standard_line.copy()
                    crit_line.kind = 'cable'
                    crit_line.quantity = 1
                    lines_changes[crit_line] = 1
                    # add node_2_3 to representatives list to not further
                    # reinforce this part off the topology in this iteration step
                    rep_main_line.append(node_2_3)
                    main_line_reinforced.append(node_2_3)

            else:
                logger.debug(
                    '==> Main line of node {} in network {} '.format(
                        repr(node), str(grid)) +
                    'has already been reinforced.')

    if main_line_reinforced:
        logger.debug('==> {} branche(s) was/were reinforced '.format(
            str(len(lines_changes))) + 'due to over-voltage issues.')

    return lines_changes


def reinforce_branches_overloading(edisgo_obj, crit_lines):
    """
    Reinforce MV or LV topology due to overloading.
    
    Parameters
    ----------
    edisgo_obj : :class:`~.edisgo.EDisGo`
    crit_lines : :pandas:`pandas.DataFrame<dataframe>`
        Dataframe containing over-loaded lines, their maximum relative
        over-loading and the corresponding time step.
        Index of the dataframe are the over-loaded lines of type
        :class:`~.network.components.Line`. Columns are 'max_rel_overload'
        containing the maximum relative over-loading as float and 'time_index'
        containing the corresponding time step the over-loading occured in as
        :pandas:`pandas.Timestamp<timestamp>`.

    Returns
    -------
    Dictionary with :class:`~.network.components.Line` and the number of Lines
    added.
        
    Notes
    -----
    Reinforce measures:

    1. Install parallel line of the same type as the existing line (Only if
       line is a cable, not an overhead line. Otherwise a standard equipment
       cable is installed right away.)
    2. Remove old line and install as many parallel standard lines as
       needed.

    """

    lines_changes = {}
    # reinforce mv lines
    lines_changes = \
        reinforce_lines_overloaded_per_grid_level(edisgo_obj, 'mv',
                                                  crit_lines, lines_changes)
    # reinforce lv lines
    lines_changes = \
        reinforce_lines_overloaded_per_grid_level(edisgo_obj, 'lv',
                                                  crit_lines, lines_changes)

    if not crit_lines.empty:
        logger.debug('==> {} branche(s) was/were reinforced '.format(
            crit_lines.shape[0]) + 'due to over-loading issues.')

    return lines_changes


def reinforce_lines_overloaded_per_grid_level(edisgo_obj, grid_level,
                                              crit_lines, lines_changes):
    def reinforce_standard_lines(relevant_lines):
        lines_standard = relevant_lines.loc[
            relevant_lines.type_info == standard_line.name]
        number_parallel_lines = np.ceil(crit_lines.max_rel_overload[
             lines_standard.index] * lines_standard.num_parallel)
        number_parallel_lines_pre = edisgo_obj.topology.lines_df.loc[
            lines_standard.index, 'num_parallel']
        edisgo_obj.topology.lines_df.loc[
            lines_standard.index, 'num_parallel'] = number_parallel_lines
        edisgo_obj.topology.lines_df.loc[
            lines_standard.index, 'x'] = edisgo_obj.topology.lines_df.loc[
            lines_standard.index, 'x'] * number_parallel_lines_pre / \
            number_parallel_lines
        edisgo_obj.topology.lines_df.loc[lines_standard.index, 'r'] = \
            edisgo_obj.topology.lines_df.loc[lines_standard.index, 'r'] * \
            number_parallel_lines_pre / number_parallel_lines
        lines_changes.update(
            (number_parallel_lines - number_parallel_lines_pre).to_dict())
        lines_default = relevant_lines.loc[
            ~relevant_lines.index.isin(lines_standard.index)]
        return lines_default

    def reinforce_single_lines(lines_default):
        lines_single = \
            lines_default.loc[lines_default.num_parallel == 1].loc[
                lines_default.kind == 'cable'].loc[
                crit_lines.max_rel_overload < 2]
        edisgo_obj.topology.lines_df.loc[
            lines_single.index, 'num_parallel'] = 2
        edisgo_obj.topology.lines_df.loc[lines_single.index, 'r'] = \
            edisgo_obj.topology.lines_df.loc[lines_single.index, 'r'] / 2
        edisgo_obj.topology.lines_df.loc[lines_single.index, 'x'] = \
            edisgo_obj.topology.lines_df.loc[lines_single.index, 'x'] / 2
        lines_changes.update({_: 1 for _ in lines_single.index})
        lines_default = lines_default.loc[
            ~lines_default.index.isin(lines_single.index)]
        return lines_default

    def reinforce_default_lines(lines_default):
        number_parallel_lines = np.ceil(lines_default.s_nom * crit_lines.loc[
            lines_default.index, 'max_rel_overload'] / (math.sqrt(3) *
            standard_line.U_n * standard_line.I_max_th))
        edisgo_obj.topology.lines_df.loc[
            lines_default.index, 'type_info'] = standard_line.name
        edisgo_obj.topology.lines_df.loc[
            lines_default.index, 's_nom'] = math.sqrt(
            3) * standard_line.U_n * standard_line.I_max_th
        edisgo_obj.topology.lines_df.loc[
            lines_default.index, 'num_parallel'] = number_parallel_lines
        edisgo_obj.topology.lines_df.loc[lines_default.index, 'r'] = \
            standard_line.R_per_km * \
            edisgo_obj.topology.lines_df.loc[lines_default.index, 'length'] / \
            edisgo_obj.topology.lines_df.loc[lines_default.index, 'num_parallel']
        omega = 2 * np.pi * edisgo_obj.config['network_parameters']['freq']
        edisgo_obj.topology.lines_df.loc[lines_default.index, 'x'] = \
            standard_line.L_per_km * omega * 1e-3 * \
            edisgo_obj.topology.lines_df.loc[lines_default.index, 'length'] / \
            edisgo_obj.topology.lines_df.loc[lines_default.index,
                                             'num_parallel']
        lines_changes.update(number_parallel_lines.to_dict())

    # load standard line data
    try:
        standard_line = \
        edisgo_obj.equipment_data['{}_cables'.format(grid_level)].loc[
            edisgo_obj.config['grid_expansion_standard_equipment'][
                '{}_line'.format(grid_level)]]
        # Todo: check voltage of standard line to distinguish between 10
        #  and 20 kV. Remove following part afterwards.
        standard_line.U_n = edisgo_obj.topology.mv_grid.nominal_voltage
    except KeyError:
        print('Chosen standard {} line is not in equipment list.'.format(
            grid_level))
    # chose lines of right grid level
    relevant_lines = edisgo_obj.topology.lines_df.loc[
        crit_lines[crit_lines.grid_level == grid_level].index]
    # handling of standard lines
    lines_default = reinforce_standard_lines(relevant_lines)
    # handling of cables where adding one cable is sufficient
    lines_default = reinforce_single_lines(lines_default)
    # default lines that haven't been handled so far
    # Todo: removed lines are not handled here unlike for trafos.
    #  Overthink and unify
    reinforce_default_lines(lines_default)

    return lines_changes
