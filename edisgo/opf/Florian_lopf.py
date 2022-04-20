# Methods to perform linearised DistFlow
import os as os
from time import perf_counter
from copy import deepcopy
import numpy as np
import pandas as pd
import pyomo.environ as pm
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.util.infeasible import log_infeasible_constraints
import itertools
import pickle
import logging

from edisgo.tools.tools import get_nodal_residual_load, calculate_impedance_for_parallel_components

BANDS = ["upper_power", "upper_energy", "lower_energy"]


# def import_flexibility_bands(dir, use_cases):
#     flexibility_bands = {}
#
#     for band in BANDS:
#         band_df = pd.DataFrame()
#         for use_case in use_cases:
#             flexibility_bands_tmp = \
#                 pd.read_csv(dir+'/{}_{}.csv'.format(band, use_case),
#                             index_col=0, parse_dates=True, dtype=np.float32)
#             band_df = pd.concat([band_df, flexibility_bands_tmp],
#                                 axis=1)
#         if band_df.columns.duplicated().any():
#             raise ValueError("Charging points with the same name in flexibility bands. "
#                              "Please check")
#         flexibility_bands[band] = band_df
#         # remove numeric problems
#         if "upper" in band:
#             flexibility_bands[band] = flexibility_bands[band] + 1e-5
#         elif "lower" in band:
#             flexibility_bands[band] = flexibility_bands[band] - 1e-5
#     return flexibility_bands


def prepare_time_invariant_parameters(edisgo, downstream_nodes_matrix, pu=True,
                                      optimize_storage=True, optimize_ev_charging=True, battery_optimized_ev_charging=True, **kwargs):
    """
    Prepare parameters that do not change within the iterations of the rolling horizon approach.
    These include topological parameters and timeseries of the inflexible units which
    are not influenced by the optimisation.

    """
    t1 = perf_counter()
    parameters = {}
    # set grid and edisgo objects as well as slack
    parameters['edisgo_object'], parameters['grid_object'], parameters['slack'] = \
        setup_grid_object(edisgo)
    parameters['downstream_nodes_matrix'] = downstream_nodes_matrix
    if optimize_storage:
        parameters['optimized_storage_units'] = \
            kwargs.get('flexible_storage_units',
                       parameters['grid_object'].storage_units_df.index)
        parameters['inflexible_storage_units'] = parameters['grid_object'].storage_units_df.index.drop(
            parameters['optimized_storage_units'])
    if optimize_ev_charging:
        parameters['ev_flex_bands'] = kwargs.get('ev_flex_bands')
        parameters['optimized_charging_points'] = parameters['ev_flex_bands']["upper_power"].columns
        parameters['inflexible_charging_points'] = parameters['grid_object'].charging_points_df.index.drop(
            parameters['optimized_charging_points'])
    # extract residual load of non optimised components
    parameters['res_load_inflexible_units'] = get_residual_load_of_not_optimized_components(
        parameters['grid_object'], parameters['edisgo_object'],
        relevant_storage_units=parameters.get('inflexible_storage_units',
                                              parameters['grid_object'].storage_units_df.index),
        relevant_charging_points=parameters.get('inflexible_charging_points',
                                                parameters['grid_object'].charging_points_df.index)
    )
    if battery_optimized_ev_charging:
        parameters['ev_battery_charging_points'] = parameters['edisgo_object'].electromobility.integrated_charging_parks_df['edisgo_id']
    # get nodal active and reactive powers of non optimised components
    # Todo: add handling of storage once become relevant
    nodal_active_power, nodal_reactive_power, nodal_active_load, nodal_reactive_load, \
    nodal_active_generation, nodal_reactive_generation, nodal_active_charging_points, \
    nodal_reactive_charging_points, nodal_active_storage, nodal_reactive_storage = get_nodal_residual_load(
        parameters['grid_object'], parameters['edisgo_object'],
        considered_storage=parameters.get('inflexible_storage_units',
                                          parameters['grid_object'].storage_units_df.index),
        considered_charging_points=parameters.get('inflexible_charging_points',
                                                  parameters['grid_object'].charging_points_df.index))
    parameters['nodal_active_power'] = nodal_active_power.T
    parameters['nodal_reactive_power'] = nodal_reactive_power.T
    parameters['nodal_active_load'] = nodal_active_load.T + nodal_active_charging_points.T
    parameters['nodal_reactive_load'] = nodal_reactive_load.T
    parameters['nodal_active_feedin'] = nodal_active_generation.T
    parameters['nodal_reactive_feedin'] = nodal_reactive_generation.T
    parameters['tan_phi_load'] = (nodal_reactive_load.divide(nodal_active_load)).fillna(0)
    parameters['tan_phi_feedin'] = (nodal_reactive_generation.divide(nodal_active_generation)).fillna(0)
    # get underlying branch elements and power factors
    # handle pu conversion
    if pu:
        print('Optimisation in pu-system. Make sure the inserted energy '
              'bands are also converted to the same pu-system.')
        parameters['v_nom'] = 1.0
        s_base = kwargs.get("s_base", 1)
        parameters['grid_object'].convert_to_pu_system(s_base, timeseries_inplace=True)
        parameters['pars'] = {'r': 'r_pu', 'x': 'x_pu', 's_nom': 's_nom_pu',
                      'p_nom': 'p_nom_pu', 'peak_load': 'peak_load_pu',
                      'capacity': 'capacity_pu'}
    else:
        parameters['v_nom'] = parameters['grid_object'].buses_df.v_nom.iloc[0]
        parameters['pars'] = {'r': 'r', 'x': 'x', 's_nom': 's_nom',
                      'p_nom': 'p_nom', 'peak_load': 'peak_load',
                      'capacity': 'capacity'}
        parameters['grid_object'].transformers_df['r'] = parameters['grid_object'].transformers_df[
                                               'r_pu'] * np.square(
            parameters['v_nom']) / parameters['grid_object'].transformers_df.s_nom
        parameters['grid_object'].transformers_df['x'] = parameters['grid_object'].transformers_df[
                                               'x_pu'] * np.square(
            parameters['v_nom']) / parameters['grid_object'].transformers_df.s_nom
    parameters['branches'] = concat_parallel_branch_elements(parameters['grid_object'])
    parameters['underlying_branch_elements'], parameters['power_factors'] = get_underlying_elements(parameters)  # Todo: time invariant
    print('It took {} seconds to extract timeinvariant parameters.'.format(perf_counter()-t1))
    return parameters


def setup_grid_object(object):
    """
    Set up the grid and edisgo object.
    """
    if hasattr(object, 'topology'):
        grid_object = deepcopy(object.topology)
        edisgo_object = deepcopy(object)
        slack = grid_object.mv_grid.station.index
    else:
        grid_object = deepcopy(object)
        edisgo_object = deepcopy(object.edisgo_obj)
        slack = [grid_object.transformers_df.bus1.iloc[
                     0]]  # Todo: careful with MV grid, does not work with that right?
    return edisgo_object, grid_object, slack


def concat_parallel_branch_elements(grid_object):
    """
    Method to merge parallel lines and transformers into one element, respectively.

    Parameters
    ----------
    grid_object

    Returns
    -------

    """
    lines = fuse_parallel_branches(grid_object.lines_df)
    trafos = grid_object.transformers_df.loc[
        grid_object.transformers_df.bus0.isin(grid_object.buses_df.index)].loc[
        grid_object.transformers_df.bus1.isin(grid_object.buses_df.index)]
    transformers = fuse_parallel_branches(trafos)
    return pd.concat([lines, transformers], sort=False)


def fuse_parallel_branches(branches):
    branches_tmp = branches[['bus0', 'bus1']]
    parallel_branches = pd.DataFrame(columns=branches.columns)
    if branches_tmp.duplicated().any():
        duplicated_branches = branches_tmp.loc[branches_tmp.duplicated(keep=False)]
        duplicated_branches['visited'] = False
        branches_tmp.drop(duplicated_branches.index, inplace=True)
        for name, buses in duplicated_branches.iterrows():
            if duplicated_branches.loc[name, 'visited']:
                continue
            else:
                parallel_branches_tmp = duplicated_branches.loc[(duplicated_branches == buses).all(axis=1)]
                duplicated_branches.loc[parallel_branches_tmp.index, 'visited'] = True
                name_par = '_'.join(str.split(name, '_')[:-1])
                parallel_branches.loc[name_par] = branches.loc[name]
                parallel_branches.loc[name_par, ['r', 'x', 's_nom']] = calculate_impedance_for_parallel_components(
                    branches.loc[parallel_branches_tmp.index, ['r', 'x', 's_nom']],
                    pu=False)
    fused_branches = pd.concat([branches.loc[branches_tmp.index], parallel_branches], sort=False)
    return fused_branches


def get_underlying_elements(parameters):
    def _get_underlying_elements(downstream_elements, power_factors, parameters, branch):
        bus0 = parameters['branches'].loc[branch, 'bus0']
        bus1 = parameters['branches'].loc[branch, 'bus1']
        s_nom = parameters['branches'].loc[branch, parameters['pars']['s_nom']]
        relevant_buses_bus0 = \
            parameters['downstream_nodes_matrix'].loc[bus0][
                parameters['downstream_nodes_matrix'].loc[bus0] == 1].index.values
        relevant_buses_bus1 = \
            parameters['downstream_nodes_matrix'].loc[bus1][
                parameters['downstream_nodes_matrix'].loc[bus1] == 1].index.values
        relevant_buses = list(set(relevant_buses_bus0).intersection(
            relevant_buses_bus1))
        downstream_elements.loc[branch, 'buses'] = relevant_buses
        if (parameters['nodal_reactive_power'].loc[relevant_buses].sum().divide(s_nom).apply(abs) > 1).any():
            print('Careful: Reactive power already exceeding line capacity for branch {}.'.format(branch))
        power_factors.loc[branch] = (1-
             parameters['nodal_reactive_power'].loc[relevant_buses].sum().divide(s_nom).apply(np.square)).apply(np.sqrt)
        if 'optimized_storage_units' in parameters:
            downstream_elements.loc[branch, 'flexible_storage'] = \
                parameters['grid_object'].storage_units_df.loc[
                    parameters['grid_object'].storage_units_df.index.isin(
                        parameters['optimized_storage_units']) &
                    parameters['grid_object'].storage_units_df.bus.isin(relevant_buses)].index.values
        else:
            downstream_elements.loc[branch, 'flexible_storage'] = []
        if 'optimized_charging_points' in parameters:
            downstream_elements.loc[branch, 'flexible_ev'] = \
                parameters['grid_object'].charging_points_df.loc[
                    parameters['grid_object'].charging_points_df.index.isin(
                        parameters['optimized_charging_points']) &
                    parameters['grid_object'].charging_points_df.bus.isin(relevant_buses)].index.values
        else:
            downstream_elements.loc[branch, 'flexible_ev'] = []
        if 'ev_battery_charging_points' in parameters:
            downstream_elements.loc[branch, 'ev_battery_charging_points'] = \
                parameters['grid_object'].charging_points_df.loc[
                    parameters['grid_object'].charging_points_df.index.isin(
                        parameters['ev_battery_charging_points']) &
                    parameters['grid_object'].charging_points_df.bus.isin(relevant_buses)].index.values
        else:
            downstream_elements.loc[branch, 'ev_battery_charging_points'] = []

        return downstream_elements, power_factors

    downstream_elements = pd.DataFrame(index=parameters['branches'].index,
                                       columns=['buses', 'flexible_storage', 'flexible_ev', 'ev_battery_charging_points'])
    power_factors = pd.DataFrame(index=parameters['branches'].index, columns=parameters['nodal_active_power'].columns)
    for branch in downstream_elements.index:
        downstream_elements, power_factors = _get_underlying_elements(downstream_elements, power_factors, parameters, branch)
    if power_factors.isna().any().any():
        print('WARNING: Branch {} is overloaded with reactive power. Still needs handling.'.format(branch))
        power_factors = power_factors.fillna(0) # Todo: ask Gaby and Birgit about this
    return downstream_elements, power_factors


def get_residual_load_of_not_optimized_components(grid, edisgo, relevant_storage_units=None,
                                                  relevant_charging_points=None, relevant_generators=None,
                                                  relevant_loads=None):
    """
    Method to get residual load of fixed components.

    Parameters
    ----------
    grid
    edisgo
    relevant_storage_units
    relevant_charging_points
    relevant_generators
    relevant_loads

    Returns
    -------

    """
    if relevant_loads is None:
        relevant_loads = grid.loads_df.index
    if relevant_generators is None:
        relevant_generators = grid.generators_df.index
    if relevant_storage_units is None:
        relevant_storage_units = grid.storage_units_df.index
    if relevant_charging_points is None:
        relevant_charging_points = grid.charging_points_df.index

    if edisgo.timeseries.charging_points_active_power.empty:
        return (
                edisgo.timeseries.generators_active_power[
                    relevant_generators].sum(axis=1)
                + edisgo.timeseries.storage_units_active_power[
                    relevant_storage_units].sum(axis=1)
                - edisgo.timeseries.loads_active_power[relevant_loads
                ].sum(axis=1)
        ).loc[edisgo.timeseries.timeindex]
    else:
        return (
                edisgo.timeseries.generators_active_power[
                    relevant_generators].sum(axis=1)
                + edisgo.timeseries.storage_units_active_power[
                    relevant_storage_units].sum(axis=1)
                - edisgo.timeseries.loads_active_power[relevant_loads
                ].sum(axis=1)
                - edisgo.timeseries.charging_points_active_power[
                    relevant_charging_points].sum(axis=1)
        ).loc[edisgo.timeseries.timeindex]


def setup_model(timeinvariant_parameters, timesteps, optimize_storage=True,
                optimize_ev_charging=True, optimize_battery_based_ev_charging=True, objective='curtailment', **kwargs):
    """
    Method to set up pyomo model for optimisation of storage procurement
    and/or ev charging with linear approximation of power flow from
    eDisGo-object.

    :param timeinvariant_parameters: parameters that stay the same for every iteration
    :param timesteps:
    :param optimize_storage:
    :param optimize_ev_charging:
    :param objective: choose the objective that should be minimized, so far
            'curtailment' and 'peak_load' are implemented
    :param kwargs:
    :return:
    """
    def init_active_nodal_power(model, bus, time):                                                                      #functions later used to define the paramter values
        return timeinvariant_parameters['nodal_active_power'].T.loc[model.timeindex[time]].loc[bus]

    def init_reactive_nodal_power(model, bus, time):
        return timeinvariant_parameters['nodal_reactive_power'].T.loc[model.timeindex[time]].loc[bus]

    def init_active_nodal_load(model, bus, time):
        return timeinvariant_parameters['nodal_active_load'].T.loc[model.timeindex[time]].loc[bus]

    def init_reactive_nodal_load(model, bus, time):
        return timeinvariant_parameters['nodal_reactive_load'].T.loc[model.timeindex[time]].loc[bus]

    def init_active_nodal_feedin(model, bus, time):
        return timeinvariant_parameters['nodal_active_feedin'].T.loc[model.timeindex[time]].loc[bus]

    def init_reactive_nodal_feedin(model, bus, time):
        return timeinvariant_parameters['nodal_reactive_feedin'].T.loc[model.timeindex[time]].loc[bus]

    def init_power_factors(model, branch, time):
        return timeinvariant_parameters['power_factors'].loc[branch, model.timeindex[time]]

    t1 = perf_counter()
    model = pm.ConcreteModel()                                                                                          #initializes the model as concrete model
    # check if correct value of objective is inserted
    if objective not in ['curtailment', 'peak_load', 'minimize_energy_level',                                           #checks if objective is valid or not implemented yet
                         'residual_load', 'maximize_energy_level', 'minimize_loading', 'dummy']:
        raise ValueError('The objective you inserted is not implemented yet.')

    edisgo_object, grid_object, slack = timeinvariant_parameters['edisgo_object'], \
                                        timeinvariant_parameters['grid_object'], \
                                        timeinvariant_parameters['slack']                                               #assignes the values from the parameter dict to new variabels
    # check if multiple voltage levels are present
    if len(grid_object.buses_df.v_nom.unique()) > 1:                                                                    #unique returns the number of voltage levels. If higher than one, different levels are present
        print('More than one voltage level included. Please make sure to '
              'adapt all impedance values to one reference system.')

    # Todo: Extract kwargs values from cfg?

    # DEFINE SETS AND FIX PARAMETERS
    print('Setup model: Defining sets and parameters.')
    model.bus_set = pm.Set(initialize=grid_object.buses_df.index)                                                       #initializes the sets with values from the timeinvariant parameters
    model.slack_bus = pm.Set(initialize=slack)                                                                          #one bus is set as slack
    model.time_set = pm.RangeSet(0, len(timesteps)-1)                                                                   #set with values from 0 to length of timesteps-1
    model.time_zero = [model.time_set.at(1)]                                                                            #first value of the time_set set
    overlap_interations = kwargs.get('overlap_interations', None)                                                       #gets the number of overlap_iterations. If nothing is passen, it inserts None
    if overlap_interations is not None:
        model.time_end= [model.time_set.at(-overlap_interations)]                                                       #if an overlap exists, time_end is set to the the value of time_set where the overlap starts
    else:
        model.time_final = [model.time_set.at(-1)]                                                                      #if no overlap is used, time_final is set to the last value of time_set
    model.time_non_zero = model.time_set - [model.time_set.at(1)]                                                       #time_non_zero is set to the values of time_set minus the first value of time_set
    model.times_fixed_soc = pm.Set(initialize=[model.time_set.at(1),                                                    #the soc is fixed for the first and the last value of the iteration
                                               model.time_set.at(-1)])
    model.timeindex = pm.Param(model.time_set, initialize={i:timesteps[i] for i in model.time_set},                     #timeindex is initialized as parameter. It is equivalent to the time_set but with date values instead of integers
                               within=pm.Any, mutable=True)
    model.time_increment = pd.infer_freq(timesteps)                                                                     #gets the frequency of the timesteps 15T
    if not any(char.isdigit() for char in model.time_increment):                                                        #if the increment is not a digit, 1 is added
        model.time_increment = '1' + model.time_increment
    model.grid = grid_object
    model.integrated_charging_parks_df = timeinvariant_parameters['edisgo_object'].electromobility.integrated_charging_parks_df
    model.EV_data = timeinvariant_parameters['edisgo_object'].electromobility.EV_dict
    if optimize_storage:
        model.storage_set = \
            pm.Set(initialize=grid_object.storage_units_df.index)
        model.optimized_storage_set = \
            pm.Set(initialize=timeinvariant_parameters['optimized_storage_units'])
        model.fixed_storage_set = model.storage_set - \
                                  model.optimized_storage_set
        model.fix_relative_soc = kwargs.get('fix_relative_soc', 0.5)
    # Todo: Define the necessary Sets for EV integration
    if optimize_battery_based_ev_charging:
        # SETS
        # -------------------
        model.car_ID_set = pm.Set(
            initialize=model.EV_data['car_id'],
            domain=pm.Integers,
            doc='Contains an ID for each car'
        )
        # Todo: why NonNegativeReals? Should these not also be integers?
        model.charger_ID_set = pm.Set(
            initialize=model.EV_data['charger_id_set'],
            domain=pm.NonNegativeReals,
            doc='Contains an ID for each charging station.',
        )
        # model.location_set = pm.Set(
        #     initialize=EV_data['location_set'],
        #     doc='Contains the location types.'
        # )
        # model.type_set = pm.Set(
        #     initialize=EV_data['type_set'],
        #     doc='Contains each car type.'
        # )

        # PARAMETERS
        # -------------------
        model.BigM = pm.Param(
            default=10 ** 6,
            doc="Big M: Sufficiently large value"
        )

        model.SOC_min_ev = pm.Param(
            model.time_set,
            model.car_ID_set,
            initialize=set_SOC_min_ev,
            mutable=True,
            doc='Defines the minimum necessary SOC for every car c and timestep t'
        )

        model.Initial_SOC_ev = pm.Param(
            model.car_ID_set,
            initialize=0,
            mutable=True,
            doc='Defines the initial SOC for every car c at the beginning of the optimization'
        )

        model.CAP_ev = pm.Param(
            model.car_ID_set,
            initialize=set_CAP_ev,
        doc = 'Defines the battery capacity'
        )

        model.Charging_station_timeseries = pm.Param(
            model.time_set,
            model.charger_ID_set,
            initialize=set_Charging_station_timeseries,
            mutable=True,
            within=pm.Any,
        doc = 'Indicates which car c is parked at the charging park s at time t.'
        )
        # model.Y_standing_ev = pm.Param(
        #     model.time_set,
        #     model.car_ID_set,
        #     model.location_set,
        #     initialize=,
        #     mutable=True,
        # doc = 'Indicates if the car is parked at one of the locations.'
        # )
        model.SOC_V2G_ev = pm.Param(
            initialize=model.EV_data['SOC_V2G'],
            mutable=True,
            doc = 'Minimum necessary SOC to allow V2G service.'
        )

        model.P_max_ev = pm.Param(
            model.car_ID_set,
            initialize=set_P_max_ev,
            doc='Maximum charging power for car car_ID .'
        )

        model.P_max_station = pm.Param(
            model.charger_ID_set,
            initialize=set_P_max_station,
            doc='Maximum charging power for charging station charger_ID.'
        )
        model.charging_efficiency = pm.Param(
            initialize=model.EV_data['charging_efficiency'],
            doc='Defines the charging and discharging efficiency (same for all cars).'
        )

        model.Consumption = pm.Param(
            model.time_set,
            model.car_ID_set,
            initialize=set_Consumption,
            default=0,
            mutable=True,
            doc='Defines the consumption for each car in each timestep.'
        )


    if optimize_ev_charging:
        model.charging_points_set = \
            pm.Set(initialize=grid_object.charging_points_df.index)
        model.flexible_charging_points_set = \
            pm.Set(initialize=timeinvariant_parameters['optimized_charging_points'])
        model.inflexible_charging_points_set = \
            model.charging_points_set - model.flexible_charging_points_set
        model.upper_ev_power = timeinvariant_parameters["ev_flex_bands"]["upper_power"]
        model.upper_ev_energy = timeinvariant_parameters["ev_flex_bands"]["upper_energy"]
        model.lower_ev_energy = timeinvariant_parameters["ev_flex_bands"]["lower_energy"]
        model.charging_efficiency = kwargs.get("charging_efficiency", 0.9)
        model.lower_bound_ev = pm.Param(model.flexible_charging_points_set, model.time_set,
                                        initialize=set_lower_band_ev, mutable=True)
        model.upper_bound_ev = pm.Param(model.flexible_charging_points_set, model.time_set,
                                        initialize=set_upper_band_ev, mutable=True)
        model.power_bound_ev = pm.Param(model.flexible_charging_points_set, model.time_set,
                                        initialize=set_power_band_ev, mutable=True)
    model.v_min = kwargs.get("v_min", 0.9)                                                                              #sets the minimal pu voltage. If nothing is passed, 0.9 is set
    model.v_max = kwargs.get("v_max", 1.1)
    model.v_nom = timeinvariant_parameters['v_nom']                                                                     #sets nominal voltage, here 10
    model.thermal_limit = kwargs.get('thermal_limit', 1.0)
    model.pars = timeinvariant_parameters['pars']
    res_load = {i: timeinvariant_parameters['res_load_inflexible_units'][model.timeindex[i]]
                for i in model.time_set}
    model.residual_load = pm.Param(model.time_set, initialize=res_load, mutable=True)                                   #sets the parameter for the residual load for every timestep
    model.downstream_nodes_matrix = timeinvariant_parameters['downstream_nodes_matrix']                                 #gives grid and downstream nodes matrix as attributes for the model

    #time_dict = {timesteps[i]: list(model.time_set.value)[i] for i in range(len(timesteps))}
    model.nodal_active_power = pm.Param(model.bus_set, model.time_set, initialize=init_active_nodal_power,              #nodal parameters are set for every timeset and every busbar
                                        mutable=True)
    model.nodal_reactive_power = pm.Param(model.bus_set, model.time_set, initialize=init_reactive_nodal_power,
                                          mutable=True)
    model.nodal_active_load = pm.Param(model.bus_set, model.time_set, initialize=init_active_nodal_load,
                                       mutable=True)
    model.nodal_reactive_load = pm.Param(model.bus_set, model.time_set, initialize=init_reactive_nodal_load,
                                         mutable=True)
    model.nodal_active_feedin = pm.Param(model.bus_set, model.time_set, initialize=init_active_nodal_feedin,
                                         mutable=True)
    model.nodal_reactive_feedin = pm.Param(model.bus_set, model.time_set, initialize=init_reactive_nodal_feedin,
                                           mutable=True)
    model.tan_phi_load = timeinvariant_parameters['tan_phi_load']
    model.tan_phi_feedin = timeinvariant_parameters['tan_phi_feedin']
    model.v_slack = kwargs.get('v_slack', model.v_nom)
    model.branches = timeinvariant_parameters['branches']
    model.branch_set = pm.Set(initialize=model.branches.index)
    model.underlying_branch_elements = timeinvariant_parameters['underlying_branch_elements']
    model.power_factors = pm.Param(model.branch_set, model.time_set, initialize=init_power_factors,
                                   mutable=True)


    if objective == 'peak_load':
        model.delta_min = kwargs.get('delta_min', 0.9)
        model.delta_max = kwargs.get('delta_max', 0.1)
        model.min_load_factor = pm.Var()
        model.max_load_factor = pm.Var()
    elif objective == 'minimize_energy_level' or \
            objective == 'maximize_energy_level':
        model.grid_power_flexible = pm.Var(model.time_set)

    # add n-1 security
    # adapt i_lines_allowed for radial feeders
    buses_in_cycles = list(
        set(itertools.chain.from_iterable(edisgo_object.topology.rings)))

    # Find lines in cycles
    lines_in_cycles = list(
        grid_object.lines_df.loc[grid_object.lines_df[[
            'bus0', 'bus1']].isin(buses_in_cycles).all(
            axis=1)].index.values)

    model.branches_load_factors = pd.DataFrame(index=model.time_set,
                                               columns=model.branch_set)
    model.branches_load_factors.loc[:, :] = 1
    tmp_residual_load = edisgo_object.timeseries.residual_load.loc[timesteps]
    indices = pd.DataFrame(index=timesteps, columns=['index'])
    indices['index'] = [i for i in range(len(timesteps))]
    model.branches_load_factors.loc[
        indices.loc[tmp_residual_load.loc[timesteps] < 0].values.T[0],
        lines_in_cycles
    ] = kwargs.get('load_factor_rings', 1.0) #0.5

    # DEFINE VARIABLES
    print('Setup model: Defining variables.')
    model.p_cum = pm.Var(model.branch_set, model.time_set)
    model.slack_p_cum_pos = pm.Var(model.branch_set, model.time_set, bounds=(0, None))
    model.slack_p_cum_neg = pm.Var(model.branch_set, model.time_set, bounds=(0, None))
    model.q_cum = pm.Var(model.branch_set, model.time_set)
    model.v = pm.Var(model.bus_set, model.time_set)
    model.slack_v_pos = pm.Var(model.bus_set, model.time_set, bounds=(0, None))
    model.slack_v_neg = pm.Var(model.bus_set, model.time_set, bounds=(0, None))
    # if not objective == 'minimize_energy_level' and \
    #         not objective == 'maximize_energy_level':
    model.curtailment_load = pm.Var(model.bus_set, model.time_set,
                                    bounds=lambda m, b, t:
                                    (0, m.nodal_active_load[b, t]))
    model.curtailment_feedin = pm.Var(model.bus_set, model.time_set,
                                      bounds=lambda m, b, t:
                                      (0, m.nodal_active_feedin[b, t]))
    if optimize_storage:
        model.soc = \
            pm.Var(model.optimized_storage_set, model.time_set,
                   bounds=lambda m, b, t: (
                   0, m.grid.storage_units_df.loc[b, model.pars['capacity']]))
        model.charging = \
            pm.Var(model.optimized_storage_set, model.time_set,
                   bounds=lambda m, b, t: (
                   -m.grid.storage_units_df.loc[b, model.pars['p_nom']],
                   m.grid.storage_units_df.loc[b, model.pars['p_nom']]))
    if optimize_battery_based_ev_charging:
        # Variables
        # -------------------
        model.soc_ev = pm.Var(
            model.time_set,
            model.car_ID_set,
            within=pm.NonNegativeReals,
        doc = 'Defines the current SOC'
        )
        model.p_out_ev = pm.Var(
            model.time_set,
            model.car_ID_set,
            within=pm.NonNegativeReals,
        doc = 'Defines the power going out of the battery' # Todo: driving/consuming? Discharging sounds like V2G
        )
        model.p_in_ev = pm.Var(
            model.time_set,
            model.car_ID_set,
            within=pm.NonNegativeReals,
            doc='Defines the power going into the battery'
        )
        model.p_station = pm.Var(
            model.time_set,
            model.charger_ID_set,
            within=pm.Reals,
            doc='Defines the power at the charging station'
        )
        model.y_V2G_ev = pm.Var(
            model.time_set,
            model.car_ID_set,
            within=pm.Binary,
            doc='Binary defining if the car is allowed to perform V2G'
        )
        model.y_in_ev = pm.Var(
            model.time_set,
            model.car_ID_set,
            within=pm.Binary,
            doc='Binary defining if the car is charging'
        )
        model.y_out_ev = pm.Var(
            model.time_set,
            model.car_ID_set,
            within=pm.Binary,
            doc='Binary defining if the car is discharging' # Todo: driving/consuming?
        )
        model.p_V2G_ev = pm.Var(
            model.time_set,
            model.car_ID_set,
            within=pm.NonNegativeReals,
            doc='Defines the power going out of the battery for V2G applications'
        )
        model.curtailment_ev = pm.Var(
            model.bus_set,
            model.time_set,
            bounds=(0, None),
        doc='The active power curtailment of ev charging')

    if optimize_ev_charging:
    # Todo: Define the necessary Variables for EV integration
        model.charging_ev = \
            pm.Var(model.flexible_charging_points_set, model.time_set,
                   bounds=lambda m, b, t:
                   (0, m.power_bound_ev[b,t]))

        model.curtailment_ev = pm.Var(model.bus_set, model.time_set,
                                      bounds=(0, None))
        if not (objective == 'minimize_energy_level' or \
            objective == 'maximize_energy_level'):
            model.energy_level_ev = \
                pm.Var(model.flexible_charging_points_set, model.time_set,
                       bounds=lambda m, b, t:
                       (m.lower_bound_ev[b, t],
                        m.upper_bound_ev[b, t]))

    # DEFINE CONSTRAINTS
    print('Setup model: Setting constraints.')
    model.ActivePower = pm.Constraint(model.branch_set, model.time_set,
                                      rule=active_power)
    model.UpperActive = pm.Constraint(model.branch_set, model.time_set,
                                      rule=upper_active_power)
    model.LowerActive = pm.Constraint(model.branch_set, model.time_set,
                                      rule=lower_active_power)
    # model.ReactivePower = pm.Constraint(model.branch_set, model.time_set,
    #                                     rule=reactive_power)
    model.SlackVoltage = pm.Constraint(model.slack_bus, model.time_set,
                                       rule=slack_voltage)
    model.VoltageDrop = pm.Constraint(model.branch_set, model.time_set,
                                      rule=voltage_drop)
    model.UpperVoltage = pm.Constraint(model.bus_set, model.time_set,
                                       rule=upper_voltage)
    model.LowerVoltage = pm.Constraint(model.bus_set, model.time_set,
                                       rule=lower_voltage)
    # model.UpperCurtLoad = pm.Constraint(model.bus_set, model.time_set,
    #                                     rule=upper_bound_curtailment_load)

    if optimize_battery_based_ev_charging:
        def energy_balance_rule (model, t, car_ID):
            if t == 0:
                return model.soc_ev[t, car_ID] == model.Initial_SOC_ev[car_ID] + model.charging_efficiency * model.p_in_ev[t, car_ID]/model.CAP_ev[car_ID] - 1 / model.charging_efficiency * (model.p_out_ev[t, car_ID] / model.CAP_ev[car_ID])
            else:
                return model.soc_ev[t, car_ID] == model.soc_ev[t-1, car_ID] + model.charging_efficiency * model.p_in_ev[t, car_ID]/model.CAP_ev[car_ID] - 1 / model.charging_efficiency * (model.p_out_ev[t, car_ID] / model.CAP_ev[car_ID])
        model.energy_balance_rule = pm.Constraint(
            model.time_set,
            model.car_ID_set,
            rule=energy_balance_rule,
            doc='Current SOC is define by previous SOC + inflows - outflows'
        )
        def SOC_constraint_rule_1 (model, t, car_ID):
            return model.soc_ev[t, car_ID] <= model.SOC_V2G_ev + (1 - model.SOC_V2G_ev) * model.y_V2G_ev[t, car_ID]
        model.SOC_constraint_rule_1 = pm.Constraint(
            model.time_set,
            model.car_ID_set,
            rule=SOC_constraint_rule_1,
            doc='Defines the upper bound for the SOC and activates the V2G binary variable',
        )
        def SOC_constraint_rule_2 (model, t, car_ID):
            return model.soc_ev[t, car_ID] >= model.SOC_V2G_ev * model.y_V2G_ev[t, car_ID]
        model.SOC_constraint_rule_2 = pm.Constraint(
            model.time_set,
            model.car_ID_set,
            rule=SOC_constraint_rule_2,
            doc='Defines the lower bound for the SOC and deactivates the V2G binary variable'
        )
        def SOC_min_rule (model, t, car_ID):
            return model.soc_ev[t, car_ID] >= model.SOC_min_ev[t, car_ID]
        model.SOC_min_rule = pm.Constraint(
            model.time_set,
            model.car_ID_set,
            rule=SOC_min_rule,
            doc='Defines the lower bound for the SOC'
        )
        def max_charge_rule (model, t, car_ID):
            return model.p_in_ev[t, car_ID] <= model.P_max_ev[car_ID]
        model.max_charge_rule = pm.Constraint(
            model.time_set,
            model.car_ID_set,
            rule=max_charge_rule,
            doc='Max power that the battery can be charged with'
        )
        def max_V2G_rule (model, t, car_ID):
            return model.p_V2G_ev[t, car_ID] <= model.y_V2G_ev[t, car_ID] * model.P_max_ev[car_ID]
        model.max_V2G_rule = pm.Constraint(
            model.time_set,
            model.car_ID_set,
            rule=max_V2G_rule,
            doc='Max V2G power that the battery can be discharged with. At public stations no V2G allowed'
        )
        # def max_V2G_rule (model, t, car_ID):
        #     return model.p_V2G_ev[t, car_ID] <= sum(
        #         model.Y_standing_ev[t, car_ID, loc] * model.P_max_ev[car_ID, loc] for loc in model.location if loc != 'public')
        # model.max_V2G_rule = pm.Constraint(
        #     model.time_set,
        #     model.car_ID_set,
        #     rule=max_V2G_rule,
        #     doc='Max V2G power that the battery can be discharged with. At public stations no V2G allowed'
        # )
        def charge_discharge_rule_1 (model, t, car_ID):
            return model.p_in_ev[t, car_ID] <= model.y_in_ev[t, car_ID] * model.BigM
        model.charge_discharge_rule_1 = pm.Constraint(
            model.time_set,
            model.car_ID_set,
            rule=charge_discharge_rule_1,
            doc='Ensures that car is not charged and discharged at the same time'
        )
        def charge_discharge_rule_2 (model, t, car_ID):
            return model.p_out_ev[t, car_ID] <= model.y_out_ev[t, car_ID] * model.BigM
        model.charge_discharge_rule_2 = pm.Constraint(
            model.time_set,
            model.car_ID_set,
            rule=charge_discharge_rule_2,
            doc='Ensures that car is not charged and discharged at the same time'
        )
        def charge_discharge_rule_3 (model, t, car_ID):
            return model.y_in_ev[t, car_ID] + model.y_out_ev[t, car_ID] <= 1
        model.charge_discharge_rule_3 = pm.Constraint(
            model.time_set,
            model.car_ID_set,
            rule=charge_discharge_rule_3,
            doc='Ensures that car is not charged and discharged at the same time'
        )
        def p_out_rule (model, t, car_ID):
            return model.p_out_ev[t, car_ID] == model.p_V2G_ev[t, car_ID] + model.Consumption[t, car_ID]
        model.p_out_rule = pm.Constraint(
            model.time_set,
            model.car_ID_set,
            rule=p_out_rule,
            doc='The power going out of the battery equals the sum of the consumption and the power supplied for V2G'
        )
        def P_max_station_rule_1 (model, t, charger_ID):
            return model.p_station[t, charger_ID] <= len(model.Charging_station_timeseries[t, charger_ID].value) * model.P_max_station[charger_ID]
        model.P_max_station_rule_1 = pm.Constraint(
            model.time_set,
            model.charger_ID_set,
            rule=P_max_station_rule_1,
            doc='Ensures that the charging demand at each station does not exceed the max possible power'
        )
        def P_max_station_rule_2 (model, t, charger_ID):
             return model.p_station[t, charger_ID] >= -len(model.Charging_station_timeseries[t, charger_ID].value) * model.P_max_station[charger_ID]

        model.P_max_station_rule_2 = pm.Constraint(
            model.time_set,
            model.charger_ID_set,
            rule=P_max_station_rule_2,
            doc='Ensures that the V2G feedin at each station does not exceed the max possible power'
        )
        def charging_station_rule (model, t, charger_ID):
            if model.Charging_station_timeseries[t, charger_ID].value[0] != 0:
                return model.p_station[t, charger_ID] == sum(
                    model.p_in_ev[t, car_ID] - model.p_V2G_ev[t, car_ID]
                    for car_ID in model.Charging_station_timeseries[t, charger_ID].value
                )
            else:
                return model.p_station[t, charger_ID] == 0
        model.charging_station_rule = pm.Constraint(
            model.time_set,
            model.charger_ID_set,
            rule=charging_station_rule,
            doc='Ensures that the charging demand at each station is the sum of the vehicle demands at this station'
        )
        model.UpperCurtEV = pm.Constraint(model.bus_set, model.time_set,
                                          rule=upper_bound_curtailment_ev)


    if optimize_storage:
        model.BatteryCharging = pm.Constraint(model.storage_set,
                                              model.time_non_zero, rule=soc)
        model.FixedSOC = pm.Constraint(model.storage_set,
                                       model.times_fixed_soc, rule=fix_soc)
    if optimize_ev_charging and not (objective == 'minimize_energy_level' or \
            objective == 'maximize_energy_level'):
    # Todo: Define the necessary Constraints for EV integration
        model.EVCharging = pm.Constraint(model.flexible_charging_points_set,
                                         model.time_non_zero, rule=charging_ev)
        model.UpperCurtEV = pm.Constraint(model.bus_set, model.time_set,
                                          rule=upper_bound_curtailment_ev)
        # set initial energy level
        energy_level_start = kwargs.get('energy_level_start', None)
        model.energy_level_start = pm.Param(model.flexible_charging_points_set,
                                            initialize=energy_level_start, mutable=True)
        model.slack_initial_energy_pos = pm.Var(model.flexible_charging_points_set, bounds=(0, None))
        model.slack_initial_energy_neg = pm.Var(model.flexible_charging_points_set, bounds=(0, None))
        model.InitialEVEnergyLevel = \
            pm.Constraint(model.flexible_charging_points_set, model.time_zero,
                              rule=initial_energy_level)
        model.InitialEVEnergyLevelStart = \
                pm.Constraint(model.flexible_charging_points_set, model.time_zero,
                              rule=fixed_energy_level)
        if energy_level_start is None:
            model.InitialEVEnergyLevel.deactivate()
        else:
            model.InitialEVEnergyLevelStart.deactivate()
        # set final energy level and if necessary charging power
        energy_level_end = kwargs.get('energy_level_end', None)
        model.energy_level_end = pm.Param(model.flexible_charging_points_set,
                                          initialize=energy_level_end, mutable=True)
        model.FinalEVEnergyLevelFix = \
            pm.Constraint(model.flexible_charging_points_set, model.time_end,
                          rule=fixed_energy_level)

        energy_level_beginning = kwargs.get('energy_level_beginning',
                                            None)
        if energy_level_beginning is None:
            model.energy_level_beginning = pm.Param(model.flexible_charging_points_set,
                                                    initialize=0, mutable=True)
        else:
            model.energy_level_beginning = pm.Param(model.flexible_charging_points_set,
                                                    initialize=energy_level_beginning, mutable=True)
        model.FinalEVEnergyLevelEnd = \
            pm.Constraint(model.flexible_charging_points_set, model.time_end,
                          rule=final_energy_level)
        model.FinalEVChargingPower = \
            pm.Constraint(model.flexible_charging_points_set, model.time_end,
                          rule=final_charging_power)
        if energy_level_end is None:
            model.FinalEVEnergyLevelFix.deactivate()
            model.FinalEVEnergyLevelEnd.deactivate()
            model.FinalEVChargingPower.deactivate()
        else:
            if type(energy_level_end) != bool:
                model.FinalEVEnergyLevelFix.deactivate()
            elif type(energy_level_end) == bool:
                model.FinalEVEnergyLevelEnd.deactivate()
        # set initial charging power
        charging_initial = kwargs.get('charging_start', None)
        model.charging_initial = pm.Param(model.flexible_charging_points_set,
                                          initialize=charging_initial, mutable=True)
        model.slack_initial_charging_pos = pm.Var(model.flexible_charging_points_set, bounds=(0, None))
        model.slack_initial_charging_neg = pm.Var(model.flexible_charging_points_set, bounds=(0, None))
        model.InitialEVChargingPower = \
            pm.Constraint(model.flexible_charging_points_set, model.time_zero,
                          rule=initial_charging_power)
        if charging_initial is None:
            model.InitialEVChargingPower.deactivate()

    if objective == 'minimize_energy_level' or \
            objective == 'maximize_energy_level':
        model.AggrGrid = pm.Constraint(model.time_set, rule=aggregated_power)

    # DEFINE OBJECTIVE
    print('Setup model: Setting objective.')
    if objective == 'peak_load':
        model.LoadFactorMin = pm.Constraint(model.time_set, rule=load_factor_min)
        model.LoadFactorMax = pm.Constraint(model.time_set, rule=load_factor_max)
        model.objective = pm.Objective(rule=minimize_max_residual_load,
                                       sense=pm.minimize,
                                       doc='Define objective function')
    elif objective == 'curtailment':
        model.objective = pm.Objective(rule=minimize_curtailment,
                                       sense=pm.minimize,
                                       doc='Define objective function')
    elif objective == 'minimize_energy_level':
        model.objective = pm.Objective(rule=minimize_energy_level,
                                       sense=pm.minimize,
                                       doc='Define objective function')
    elif objective == 'maximize_energy_level':
        model.objective = pm.Objective(rule=maximize_energy_level,
                                       sense=pm.minimize,
                                       doc='Define objective function')
    elif objective == 'residual_load':
        model.grid_residual_load = pm.Var(model.time_set)
        model.GridResidualLoad = pm.Constraint(model.time_set,
                                               rule=grid_residual_load)
        model.objective = pm.Objective(rule=minimize_residual_load,
                                       sense=pm.minimize,
                                       doc='Define objective function')
    elif objective == 'minimize_loading':
        model.objective = pm.Objective(rule=minimize_loading,
                                       sense=pm.minimize,
                                       doc='Define objective function')
    elif objective == 'dummy':
        model.objective = pm.Objective(rule=dummy,
                                       sense=pm.maximize,
                                       doc='Dummy objective')
    else:
        raise Exception('Unknown objective.')

    if kwargs.get('print_model', False):
        model.pprint()
    print('Successfully set up optimisation model.')
    print('It took {} seconds to set up model.'.format(perf_counter() - t1))
    return model


### Florian Additions
def set_SOC_min_ev(model, time, car_id):
    return model.EV_data['SOC_min'].loc[(model.timeindex[time], car_id)].values[0]

def set_Initial_SOC_ev(model, time, car_id):
    return model.EV_data['SOC_min'].loc[(model.timeindex[time], car_id)].values[0]

def set_CAP_ev(model, car_id):
    return model.EV_data['bat_cap'].loc[car_id].values[0][0]

def set_Charging_station_timeseries(model, time, charger_id):
    if model.timeindex[time].value in model.EV_data['charger_time_series'][charger_id].index.values:
        return model.EV_data['charger_time_series'][charger_id].loc[model.timeindex[time]]
    else:
        return [0]

def set_P_max_ev(model, car_ID):
    return model.EV_data['Car_Characteristics'].loc[
        model.EV_data['Car_Characteristics']['Nutzbare Batteriekapazitaet'] == model.EV_data['bat_cap'][1][car_ID][0]][
        'Maximale Ladeleistung'].values[0]

def set_P_max_station(model, charger_ID):
    if charger_ID == 0:
        return 0
    else:
        return model.grid.charging_points_df['p_nom'][model.integrated_charging_parks_df['edisgo_id'][charger_ID]]

def set_Consumption(model, time, car_id):
    return model.EV_data['consumption'].loc[(model.timeindex[time], car_id)].values[0]



def set_lower_band_ev(model, cp, time):
    return model.lower_ev_energy.loc[model.timeindex[time], cp]


def set_upper_band_ev(model, cp, time):
    return model.upper_ev_energy.loc[model.timeindex[time], cp]


def set_power_band_ev(model, cp, time):
    return model.upper_ev_power.loc[model.timeindex[time], cp]


def active_power(model, branch, time):
    '''
    Constraint for active power at node
    :param model:
    :param bus:
    :param time:
    :return:
    '''
    relevant_buses = model.underlying_branch_elements.loc[branch, 'buses']
    relevant_storage_units = model.underlying_branch_elements.loc[branch, 'flexible_storage']
    relevant_charging_points = model.underlying_branch_elements.loc[branch, 'flexible_ev']
    load_flow_on_line = \
        sum([model.nodal_active_power[bus, time] for bus in relevant_buses])
    if hasattr(model, 'flexible_charging_points_set'):
        return model.p_cum[branch, time] == load_flow_on_line + \
               sum(model.charging[storage, time]
                for storage in relevant_storage_units) - \
               sum(model.charging_ev[cp, time] for cp in relevant_charging_points) + \
               sum(model.curtailment_load[bus, time] + model.curtailment_ev[bus, time] -
                model.curtailment_feedin[bus, time] for bus in relevant_buses)
    elif hasattr(model, 'car_ID_set'):
        relevant_charging_points = model.underlying_branch_elements.loc[branch, 'ev_battery_charging_points']
        df = model.integrated_charging_parks_df
        return model.p_cum[branch, time] == load_flow_on_line + \
               sum(model.charging[storage, time]
                for storage in relevant_storage_units) - \
               sum(model.p_station[time, df.loc[df['edisgo_id']==cp].index.values[0]] for cp in relevant_charging_points) + \
               sum(model.curtailment_load[bus, time] + model.curtailment_ev[bus, time] -
                model.curtailment_feedin[bus, time] for bus in relevant_buses)
    else:
        return model.p_cum[branch, time] == load_flow_on_line + \
               sum(model.charging[storage, time]
                   for storage in relevant_storage_units) + \
               sum(model.curtailment_load[bus, time] -
                   model.curtailment_feedin[bus, time] for bus in relevant_buses)


def upper_active_power(model, branch, time):
    """
    Upper bound of active branch power
    """
    return model.p_cum[branch, time] <= model.thermal_limit *model.power_factors[branch, time] * \
           model.branches.loc[branch, model.pars['s_nom']] + model.slack_p_cum_pos[branch, time]


def lower_active_power(model, branch, time):
    """
    Lower bound of active branch power
    """
    return model.p_cum[branch, time] >= - model.thermal_limit *model.power_factors[branch, time] * \
           model.branches.loc[branch, model.pars['s_nom']] - model.slack_p_cum_neg[branch, time]


def slack_voltage(model, bus, time):
    """
    Constraint that fixes voltage to nominal voltage
    :param model:
    :param bus:
    :param time:
    :return:
    """
    timeindex = model.timeindex[time]
    if isinstance(model.v_slack, pd.Series):
        return model.v[bus, time] == np.square(model.v_slack[timeindex] *
                                               model.v_nom)
    else:
        return model.v[bus, time] == np.square(model.v_slack)


def voltage_drop(model, branch, time):
    """
    Constraint that describes the voltage drop over one line
    :param model:
    :param branch:
    :param time:
    :return:
    """
    bus0 = model.branches.loc[branch, 'bus0']
    bus1 = model.branches.loc[branch, 'bus1']
    if model.downstream_nodes_matrix.loc[bus0, bus1] == 1:
        upstream_bus = bus0
        downstream_bus = bus1
    elif model.downstream_nodes_matrix.loc[bus1, bus0] == 1:
        upstream_bus = bus1
        downstream_bus = bus0
    else:
        raise Exception('Something went wrong. Bus0 and bus1 of line {} are '
                        'not connected in downstream_nodes_matrix.'.format(branch))
    q_cum = get_q_line(model, branch, time)
    return model.v[downstream_bus, time] == model.v[upstream_bus, time] + \
        2 * (model.p_cum[branch, time] * model.branches.loc[branch, model.pars['r']] +
             q_cum * model.branches.loc[branch, model.pars['x']])


def get_q_line(model, branch, time, get_results=False):
    """
    Method to extract reactive power flow on line.

    :param model:
    :param branch:
    :param time:
    :return:
    """
    timeindex = model.timeindex[time]
    relevant_buses = model.underlying_branch_elements.loc[branch, 'buses']
    load_flow_on_line = \
        sum([model.nodal_reactive_power[bus, time] for bus in relevant_buses])
    if get_results:
        return (load_flow_on_line + sum(
            model.curtailment_load[bus, time].value * model.tan_phi_load.loc[timeindex, bus] - #Todo: Find out if this should be positive or negative
            model.curtailment_feedin[bus, time].value * model.tan_phi_feedin.loc[timeindex, bus] for bus in
            relevant_buses))
    else:
        return (load_flow_on_line + sum(
                 model.curtailment_load[bus, time] * model.tan_phi_load.loc[timeindex, bus] - #Todo: Find out if this should be positive or negative, analogously to q_cum
                 model.curtailment_feedin[bus, time] * model.tan_phi_feedin.loc[timeindex, bus] for bus in
                 relevant_buses))


def upper_voltage(model, bus, time):
    """
    Upper bound on voltage at buses
    """
    return model.v[bus, time] <= np.square(model.v_max * model.v_nom) + model.slack_v_pos[bus, time]


def lower_voltage(model, bus, time):
    """
    Lower bound on voltage at buses
    """
    return model.v[bus, time] >= np.square(model.v_min * model.v_nom) - model.slack_v_neg[bus, time]


def soc(model, storage, time):
    '''
    Constraint for battery charging #Todo: Check if time-1 or time for charging
    :param model:
    :param storage:
    :param time:
    :return:
    '''
    return model.soc[storage, time] == model.soc[storage, time - 1] - \
           model.grid.storage_units_df.loc[storage, 'efficiency_store'] * \
           model.charging[storage, time - 1] *\
           (pd.to_timedelta(model.time_increment)/pd.to_timedelta('1h'))


def fix_soc(model, bus, time):
    '''
    Constraint with which state of charge at beginning and end of charging
    period is fixed at certain value
    :param model:
    :param bus:
    :param time:
    :return:
    '''
    return model.soc[bus, time] == model.fix_relative_soc * \
           model.grid.storage_units_df.loc[bus, model.pars['capacity']]


def charging_ev(model, charging_point, time):
    """
    Constraint for charging of EV that has to ly between the lower and upper
    energy band. #Todo: Check if time-1 or time for charging

    :param model:
    :param charging_point:
    :param time:
    :return:
    """
    return model.energy_level_ev[charging_point, time] == \
           model.energy_level_ev[charging_point, time - 1] + \
           model.charging_efficiency * \
           model.charging_ev[charging_point, time]*\
           (pd.to_timedelta(model.time_increment)/pd.to_timedelta('1h'))


def upper_bound_curtailment_ev(model, bus, time):
    """
    Upper bound for the curtailment of flexible EVs.

    Parameters
    ----------
    model
    bus
    time

    Returns
    -------
    Constraint for optimisation
    """
    if hasattr(model, 'flexible_charging_points_set'):
        relevant_charging_points = model.grid.charging_points_df.loc[
        model.grid.charging_points_df.index.isin(model.flexible_charging_points_set) &
        model.grid.charging_points_df.bus.isin([bus])].index.values
    else:
        relevant_charging_points = model.grid.charging_points_df.loc[
        model.grid.charging_points_df.index.isin(model.integrated_charging_parks_df['edisgo_id']) &
        model.grid.charging_points_df.bus.isin([bus])].index.values
    if len(relevant_charging_points) < 1:
        return model.curtailment_ev[bus, time] <= 0
    elif hasattr(model, 'car_ID_set'):
        df = model.integrated_charging_parks_df
        return model.curtailment_ev[bus, time] <= \
               sum(model.p_station[time, df.loc[df['edisgo_id']==cp].index.values[0]] for cp in relevant_charging_points)
    else:
        return model.curtailment_ev[bus, time] <= \
               sum(model.charging_ev[cp, time] for cp in relevant_charging_points)


def initial_energy_level(model, charging_point, time):
    '''
    Constraint for initial value of energy
    :param model:
    :param charging_point:
    :param time:
    :return:
    '''
    return model.energy_level_ev[charging_point, time] == \
           model.energy_level_start[charging_point] + model.slack_initial_energy_pos[charging_point] - \
           model.slack_initial_energy_neg[charging_point]


def fixed_energy_level(model, charging_point, time):
    '''
    Constraint for initial value of energy
    :param model:
    :param charging_point:
    :param time:
    :return:
    '''
    initial_lower_band = \
        model.lower_bound_ev[charging_point, time]
    initial_upper_band = \
        model.upper_bound_ev[charging_point, time]
    return model.energy_level_ev[charging_point, time] == \
           (initial_lower_band+initial_upper_band)/2


def final_energy_level(model, charging_point, time):
    '''
    Constraint for final value of energy in last iteration
    :param model:
    :param charging_point:
    :param time:
    :return:
    '''
    return model.energy_level_ev[charging_point, time] == \
           model.energy_level_beginning[charging_point] + model.energy_level_end[charging_point]


def final_charging_power(model, charging_point, time):
    '''
    Constraint for final value of charging power, setting it to 0
    :param model:
    :param charging_point:
    :param time:
    :return:
    '''
    return model.charging_ev[charging_point, time] == 0


def initial_charging_power(model, charging_point, time):
    '''
    Constraint for initial value of charging power
    :param model:
    :param charging_point:
    :param time:
    :return:
    '''
    return model.charging_ev[charging_point, time] == \
           model.charging_initial[charging_point] + model.slack_initial_charging_pos[charging_point] - \
           model.slack_initial_charging_neg[charging_point]


def aggregated_power(model, time):
    """
    Todo: add docstring
    """
    if hasattr(model, 'storage_set'):
        relevant_storage_units = model.optimized_storage_set
    else:
        relevant_storage_units = []
    if hasattr(model, 'charging_points_set'):
        relevant_charging_points = model.flexible_charging_points_set
    else:
        relevant_charging_points = []
    return model.grid_power_flexible[time] == \
           -sum(model.charging[storage, time] for storage in relevant_storage_units)+ \
           sum(model.charging_ev[cp, time] for cp in relevant_charging_points)


def load_factor_min(model, time):
    '''
    Constraint that describes the minimum load factor.
    :param model:
    :param time:
    :return:
    '''
    if hasattr(model, 'storage_set'):
        relevant_storage_units = model.optimized_storage_set
    else:
        relevant_storage_units = []
    if hasattr(model, 'charging_points_set'):
        relevant_charging_points = model.flexible_charging_points_set
    else:
        relevant_charging_points = []
    return model.min_load_factor <= model.residual_load[time] + \
        sum(model.charging[storage, time] for storage in relevant_storage_units) - \
        sum(model.charging_ev[cp, time] for cp in relevant_charging_points)  + \
        sum(model.curtailment_load[bus, time] -
            model.curtailment_feedin[bus, time] for bus in model.bus_set)


def load_factor_max(model, time):
    '''
    Constraint that describes the maximum load factor.
    :param model:
    :param time:
    :return:
    '''
    if hasattr(model, 'storage_set'):
        relevant_storage_units = model.optimized_storage_set
    else:
        relevant_storage_units = []
    if hasattr(model, 'charging_points_set'):
        relevant_charging_points = model.flexible_charging_points_set
    else:
        relevant_charging_points = []
    return model.max_load_factor >= model.residual_load[time] + \
        sum(model.charging[storage, time] for storage in relevant_storage_units)- \
        sum(model.charging_ev[cp, time] for cp in relevant_charging_points) + \
        sum(model.curtailment_load[bus, time] -
            model.curtailment_feedin[bus, time] for bus in model.bus_set)


def minimize_max_residual_load(model):
    """
    Objective minimizing extreme load factors
    :param model:
    :return:
    """
    if hasattr(model, 'flexible_charging_points_set'):
        return -model.delta_min * model.min_load_factor + \
               model.delta_max * model.max_load_factor + \
               sum(model.curtailment_load[bus, time] +
                   model.curtailment_feedin[bus, time] +
                   0.5*model.curtailment_ev[bus,time]
                   for bus in model.bus_set
                   for time in model.time_set)
    else:
        return -model.delta_min * model.min_load_factor + \
               model.delta_max * model.max_load_factor + \
               sum(model.curtailment_load[bus, time] +
                   model.curtailment_feedin[bus, time]
                   for bus in model.bus_set
                   for time in model.time_set)


def minimize_curtailment(model):
    """
    Objective minimizing required curtailment. CAREFUL: Solution not unambiguous.
    :param model:
    :return:
    """
    if hasattr(model, 'charging_points_set') or hasattr(model, 'charger_ID_set'):
        return sum(model.curtailment_load[bus, time] +
                   model.curtailment_feedin[bus, time] +
                   0.5*model.curtailment_ev[bus,time]
                   for bus in model.bus_set
                   for time in model.time_set)
    else:
        return sum(model.curtailment_load[bus, time] +
                   model.curtailment_feedin[bus, time]
                   for bus in model.bus_set
                   for time in model.time_set)


def minimize_energy_level(model):
    """
    Objective minimizing energy level of grid while also minimizing necessary curtailment
    :param model:
    :return:
    """
    if hasattr(model, 'charging_points_set'):
        return sum(model.curtailment_load[bus, time] +
                   model.curtailment_feedin[bus, time] +
                   0.5*model.curtailment_ev[bus,time]
                   for bus in model.bus_set
                   for time in model.time_set)*1e6 + \
               sum(model.grid_power_flexible[time] for
                   time in model.time_set)
    else:
        return sum(model.curtailment_load[bus, time] +
                   model.curtailment_feedin[bus, time]
                   for bus in model.bus_set
                   for time in model.time_set) * 1e6 + \
               sum(model.grid_power_flexible[time] for
                   time in model.time_set)


def maximize_energy_level(model):
    """
    Objective maximizing energy level of grid while also minimizing necessary curtailment
    :param model:
    :return:
    """
    if hasattr(model, 'charging_points_set'):
        return sum(model.curtailment_load[bus, time] +
                   model.curtailment_feedin[bus, time] +
                   0.5*model.curtailment_ev[bus,time]
                   for bus in model.bus_set
                   for time in model.time_set)*1e6 - \
               sum(model.grid_power_flexible[time] for
                   time in model.time_set)
    else:
        return sum(model.curtailment_load[bus, time] +
                   model.curtailment_feedin[bus, time]
                   for bus in model.bus_set
                   for time in model.time_set) * 1e6 - \
               sum(model.grid_power_flexible[time] for
                   time in model.time_set)


def grid_residual_load(model, time):
    if hasattr(model, 'storage_set'):
        relevant_storage_units = model.optimized_storage_set
    else:
        relevant_storage_units = []
    if hasattr(model, 'charging_points_set'):
        relevant_charging_points = model.flexible_charging_points_set
    else:
        relevant_charging_points = []
    return model.grid_residual_load[time] == \
    model.residual_load[time] + \
        sum(model.charging[storage, time] for storage in relevant_storage_units) - \
    sum(model.charging_ev[cp, time] for cp in relevant_charging_points)  #+ \
    # sum(model.curtailment_load[bus, time] -
    #     model.curtailment_feedin[bus, time] for bus in model.bus_set)


def minimize_residual_load(model):
    """
    Objective minimizing curtailment and squared residual load
    :param model:
    :return:
    """
    if hasattr(model, 'slack_initial_charging_pos'):
        slack_charging = sum(model.slack_initial_charging_pos[cp]+model.slack_initial_charging_neg[cp]
                             for cp in model.flexible_charging_points_set)
    else:
        slack_charging = 0
    if hasattr(model, 'slack_initial_energy_pos'):
        slack_energy = sum(model.slack_initial_energy_pos[cp]+model.slack_initial_energy_neg[cp]
                             for cp in model.flexible_charging_points_set)
    else:
        slack_energy = 0
    if hasattr(model, 'charging_points_set'):
        return 1e-5*sum(model.grid_residual_load[time]**2 for time in model.time_set) + \
            sum(1e-2*(model.curtailment_load[bus, time] + model.curtailment_feedin[bus, time] +
                0.5*model.curtailment_ev[bus,time]) +
                1000* (model.slack_v_pos[bus, time] + model.slack_v_neg[bus, time])
            for bus in model.bus_set for time in model.time_set) + 1000*(slack_charging + slack_energy) + \
            1000*sum(model.slack_p_cum_pos[branch, time] + model.slack_p_cum_neg[branch, time]
                     for branch in model.branch_set for time in model.time_set)
    else:
        return 1e-5 * sum(model.grid_residual_load[time] ** 2 for time in model.time_set) + \
               sum(1e-2 * (model.curtailment_load[bus, time] + model.curtailment_feedin[bus, time]) +
                   1000 * (model.slack_v_pos[bus, time] + model.slack_v_neg[bus, time])
                   for bus in model.bus_set for time in model.time_set)  + \
               1000 * sum(model.slack_p_cum_pos[branch, time] + model.slack_p_cum_neg[branch, time]
                          for branch in model.branch_set for time in model.time_set)


def minimize_loading(model):
    """
    Objective minimizing curtailment and squared term of component loading
    :param model:
    :return:
    """
    if hasattr(model, 'slack_initial_charging_pos'):
        slack_charging = sum(model.slack_initial_charging_pos[cp]+model.slack_initial_charging_neg[cp]
                             for cp in model.flexible_charging_points_set)
    else:
        slack_charging = 0
    if hasattr(model, 'slack_initial_energy_pos'):
        slack_energy = sum(model.slack_initial_energy_pos[cp]+model.slack_initial_energy_neg[cp]
                             for cp in model.flexible_charging_points_set)
    else:
        slack_energy = 0
    if hasattr(model, 'charging_points_set'):
        return 1e-5*sum((model.p_cum[branch, time]/
                         (model.power_factors[branch, time]*model.branches.loc[branch, model.pars['s_nom']]))**2
                        for branch in model.branch_set for time in model.time_set) + \
               sum(1e-2*(model.curtailment_load[bus, time] + model.curtailment_feedin[bus, time] +
                0.5*model.curtailment_ev[bus,time]) +
                1000* (model.slack_v_pos[bus, time] + model.slack_v_neg[bus, time])
            for bus in model.bus_set for time in model.time_set) + 1000*(slack_charging + slack_energy) + \
            1000*sum(model.slack_p_cum_pos[branch, time] + model.slack_p_cum_neg[branch, time]
                     for branch in model.branch_set for time in model.time_set)
    else:
        return 1e-5*sum(model.p_cum[branch, time].divide(
            model.power_factors[branch, time]*model.branches.loc[branch, model.pars['s_nom']])**2
                        for branch in model.branch_set for time in model.time_set) + \
                sum(1e-2 * (model.curtailment_load[bus, time] + model.curtailment_feedin[bus, time]) +
                   1000 * (model.slack_v_pos[bus, time] + model.slack_v_neg[bus, time])
                   for bus in model.bus_set for time in model.time_set)  + \
               1000 * sum(model.slack_p_cum_pos[branch, time] + model.slack_p_cum_neg[branch, time]
                          for branch in model.branch_set for time in model.time_set)

def dummy(model):
    """
    Dummy objective.
    :param model:
    :return:
    """
    return sum(model.soc_ev[t, car_ID]
               for t in model.time_set
               for car_ID in model.car_ID_set)\
           + 0.1 * sum(model.curtailment_load[bus, t] + model.curtailment_feedin[bus, t] + model.curtailment_ev[bus, t]
               for bus in model.bus_set
               for t in model.time_set)


def update_model(model, timesteps, parameters, overlap_interations, optimize_storage=True, optimize_ev=True, optimize_battery_based_ev_charging=True, **kwargs):
    """
    Method to update model parameter where necessary if rolling horizon
    optimization is chosen.

    Parameters
    ----------
    model
    timesteps
    parameters
    optimize_storage
    optimize_ev
    kwargs

    Returns
    -------

    """
    print('Updating model')
    t1 = perf_counter()
    for i in model.time_set:
        overlap = i - len(timesteps) + 1
        if overlap > 0:
            timeindex = timesteps[-1] + pd.to_timedelta(model.time_increment)*overlap
            indexer = timesteps[-1]
        else:
            timeindex = timesteps[i]
            indexer = timesteps[i]
        model.timeindex[i].set_value(timeindex)
        model.residual_load[i].set_value(parameters['res_load_inflexible_units'][indexer])
        for bus in model.bus_set:
            model.nodal_active_power[bus, i].set_value(parameters['nodal_active_power'].loc[bus, indexer])
            model.nodal_reactive_power[bus, i].set_value(parameters['nodal_reactive_power'].loc[bus, indexer])
            model.nodal_active_load[bus, i].set_value(parameters['nodal_active_load'].loc[bus, indexer])
            model.nodal_reactive_load[bus, i].set_value(parameters['nodal_reactive_load'].loc[bus, indexer])
            model.nodal_active_feedin[bus, i].set_value(parameters['nodal_active_feedin'].loc[bus, indexer])
            model.nodal_reactive_feedin[bus, i].set_value(parameters['nodal_reactive_feedin'].loc[bus, indexer])

        for branch in model.branch_set:
            model.power_factors[branch, i].set_value(parameters['power_factors'].loc[branch, indexer])

    if optimize_ev:
        for t in model.time_set:
            overlap = t - len(timesteps) + 1
            if overlap > 0:
                indexer = len(timesteps)-1
            else:
                indexer = t
            for cp in model.flexible_charging_points_set:
                model.power_bound_ev[cp, t].set_value(set_power_band_ev(model, cp, indexer))
                model.lower_bound_ev[cp, t].set_value(set_lower_band_ev(model, cp, indexer))
                model.upper_bound_ev[cp, t].set_value(set_upper_band_ev(model, cp, indexer))
        # set initial energy level
        energy_level_start = kwargs.get('energy_level_start', None)
        charging_initial = kwargs.get('charging_start', None)
        # if run is new start of era deactivate initial energy level, otherwise activate initial energy and charging
        if energy_level_start is None:
            model.InitialEVEnergyLevel.deactivate()
            model.InitialEVEnergyLevelStart.activate()
        else:
            for cp in model.flexible_charging_points_set:
                model.energy_level_start[cp].set_value(energy_level_start[cp])
            model.InitialEVEnergyLevel.activate()
            model.InitialEVEnergyLevelStart.deactivate()
        # set initial charging
        if charging_initial is not None:
            for cp in model.flexible_charging_points_set:
                model.charging_initial[cp].set_value(charging_initial[cp])
            model.InitialEVChargingPower.activate()
        # set energy level beginning if necessary

        energy_level_beginning = kwargs.get('energy_level_beginning',
                                            None)
        if energy_level_beginning is not None:
            for cp in model.flexible_charging_points_set:
                model.energy_level_beginning[cp].set_value(energy_level_beginning[cp])

        # set final energy level and if necessary charging power
        energy_level_end = kwargs.get('energy_level_end', None)
        if energy_level_end is None:
            model.FinalEVEnergyLevelFix.deactivate()
            model.FinalEVEnergyLevelEnd.deactivate()
            model.FinalEVChargingPower.deactivate()
        elif type(energy_level_end) == bool:
            model.FinalEVEnergyLevelFix.activate()
            model.FinalEVEnergyLevelEnd.deactivate()
            model.FinalEVChargingPower.activate()
        else:
            for cp in model.flexible_charging_points_set:
                model.energy_level_end[cp].set_value(energy_level_end[cp])
            model.FinalEVEnergyLevelEnd.activate()
            model.FinalEVChargingPower.activate()
            model.FinalEVEnergyLevelFix.deactivate()

    if optimize_battery_based_ev_charging:
        for car_id in model.car_ID_set:
            model.Initial_SOC_ev[car_id] = model.soc_ev[model.time_set.at(-1) - overlap_interations, car_id].value
        for t in model.time_set:
            overlap = t - len(timesteps) + 1
            if overlap > 0:
                indexer = len(timesteps)-1
            else:
                indexer = t
            for car_id in model.car_ID_set:
                model.SOC_min_ev[t, car_id].set_value(set_SOC_min_ev(model, indexer, car_id))
                model.Consumption[t, car_id].set_value(set_Consumption(model, indexer, car_id))
            for charger_id in model.charger_ID_set:
                model.Charging_station_timeseries[t, charger_id].set_value(set_Charging_station_timeseries(model, indexer, charger_id))

        #Deactivate all EV constraints for debugging
        # model.SOC_constraint_rule_1.deactivate()
        # model.SOC_constraint_rule_2.deactivate()
        # model.SOC_min_rule.deactivate()
        # model.P_max_station_rule.deactivate()
        # model.UpperCurtEV.deactivate()
        # model.charge_discharge_rule_1.deactivate()
        # model.charge_discharge_rule_2.deactivate()
        # model.charge_discharge_rule_3.deactivate()
        # model.charging_station_rule.deactivate()
        # model.energy_balance_rule.deactivate()
        # model.max_V2G_rule.deactivate()
        # model.max_charge_rule.deactivate()
        #
        # model.SOC_min_rule.activate()
        # model.P_max_station_rule.activate()
        # model.UpperCurtEV.activate()
        # model.charge_discharge_rule_1.activate()
        # model.charge_discharge_rule_2.activate()
        # model.charge_discharge_rule_3.activate()
        # model.charging_station_rule.activate()
        # model.energy_balance_rule.activate()
        # model.max_V2G_rule.activate()
        # model.max_charge_rule.activate()


    if optimize_storage:
        raise NotImplementedError
    print('It took {} seconds to update the model.'.format(perf_counter() - t1))
    return model


def optimize(model, solver, load_solutions=True, mode=None):
    """
    Method to run the optimization and extract the results.

    :param model: pyomo.environ.ConcreteModel
    :param solver: str
                    Solver type, e.g. 'glpk', 'gurobi', 'ipopt'
    :param save_dir: str
                    directory to which results are saved, default None will
                    no saving of the results
    :return:
    """
    print('Starting optimisation')
    t1 = perf_counter()
    opt = pm.SolverFactory(solver)
    opt.options['threads'] = 8

    # Optimize
    results = opt.solve(model, tee=True,
                        load_solutions=load_solutions)

    if (results.solver.status == SolverStatus.ok) and \
            (
                    results.solver.termination_condition == TerminationCondition.optimal):
        print('Model Solved to Optimality')
        # Extract results
        time_dict = {t: model.timeindex[t].value for t in model.time_set}
        result_dict = {}
        if hasattr(model, 'storage_set'):
            result_dict['x_charge'] = pd.Series(model.charging.extract_values()).unstack().rename(columns=time_dict).T
            result_dict['soc'] = pd.Series(model.soc.extract_values()).unstack().rename(columns=time_dict).T
        if hasattr(model, 'flexible_charging_points_set'):
            result_dict['x_charge_ev'] = pd.Series(model.charging_ev.extract_values()).unstack().rename(
                columns=time_dict).T
            result_dict['energy_level_cp'] = pd.Series(model.energy_level_ev.extract_values()).unstack().rename(
                columns=time_dict).T
            result_dict['slack_charging'] = pd.Series(model.slack_initial_charging_pos.extract_values()) + \
                                            pd.Series(model.slack_initial_charging_neg.extract_values())
            result_dict['slack_energy'] = pd.Series(model.slack_initial_energy_pos.extract_values()) + \
                                          pd.Series(model.slack_initial_energy_neg.extract_values())
        result_dict['curtailment_load'] = pd.Series(model.curtailment_load.extract_values()).unstack().rename(
            columns=time_dict).T
        result_dict['curtailment_feedin'] = pd.Series(model.curtailment_feedin.extract_values()).unstack().rename(
            columns=time_dict).T
        if hasattr(model, 'flexible_charging_points_set'):
            result_dict['curtailment_ev'] = pd.Series(model.curtailment_ev.extract_values()).unstack().rename(
            columns=time_dict).T
        if hasattr(model, 'car_ID_set'):
            result_dict['SOC_ev'] = pd.Series(model.soc_ev.extract_values()).unstack().rename(index=time_dict)
            result_dict['curtailment_ev'] = pd.Series(model.curtailment_ev.extract_values()).unstack().rename(
                columns=time_dict).T
            result_dict['p_station'] = pd.Series(model.p_station.extract_values()).unstack().rename(
                index=time_dict)
            result_dict['p_ev'] = pd.Series(model.p_in_ev.extract_values()).unstack().rename(
                index=time_dict) - pd.Series(model.p_out_ev.extract_values()).unstack().rename(
                index=time_dict)
        result_dict['p_line'] = pd.Series(model.p_cum.extract_values()).unstack().rename(columns=time_dict).T
        result_dict['q_line'] = pd.Series(model.q_cum.extract_values()).unstack().rename(columns=time_dict).T
        result_dict['v_bus'] = pd.Series(model.v.extract_values()).unstack().rename(columns=time_dict).T.apply(np.sqrt)
        result_dict['slack_v_pos'] = pd.Series(model.slack_v_pos.extract_values()).unstack().rename(columns=time_dict).T
        result_dict['slack_v_neg'] = pd.Series(model.slack_v_neg.extract_values()).unstack().rename(columns=time_dict).T
        result_dict['slack_p_cum_pos'] = pd.Series(model.slack_p_cum_pos.extract_values()).unstack().rename(
            columns=time_dict).T
        result_dict['slack_p_cum_neg'] = pd.Series(model.slack_p_cum_pos.extract_values()).unstack().rename(
            columns=time_dict).T
        if mode == 'energy_band':
            result_dict['p_aggr'] = pd.Series(model.grid_power_flexible.extract_values()).rename(time_dict)
        # Todo: check if this works
        index = result_dict['curtailment_load'].index[result_dict['curtailment_load'].index.isin(model.tan_phi_load.index)]
        result_dict['curtailment_reactive_load'] = result_dict['curtailment_load'].multiply(
            model.tan_phi_load.loc[index, result_dict['curtailment_load'].columns]).dropna(how='all')
        result_dict['curtailment_reactive_feedin'] = result_dict['curtailment_feedin'].multiply(
            model.tan_phi_feedin.loc[index, result_dict['curtailment_feedin'].columns]).dropna(how='all')

        print('It took {} seconds to optimize model.'.format(perf_counter() - t1))
        return result_dict
    elif results.solver.termination_condition == TerminationCondition.infeasible:
        print('Model is infeasible')
        log_infeasible_constraints(model, log_expression=True, log_variables=True)
        logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.INFO)
        return
        # Do something when model in infeasible
    else:
        print('Solver Status: ', results.solver.status)
        return


def combine_results_for_grid(feeders, grid_id, res_dir, res_name):
    res_grid = pd.DataFrame()
    for feeder_id in feeders:
        res_feeder = pd.DataFrame()
        for i in range(14):
            try:
                res_feeder_tmp = pd.read_csv(res_dir + '/{}/{}/{}_{}_{}_{}.csv'.format(
                    grid_id, feeder_id, res_name, grid_id, feeder_id, i),
                                                     index_col=0, parse_dates=True)
                res_feeder = pd.concat([res_feeder, res_feeder_tmp], sort=False)
            except:
                print('Results for feeder {} in grid {} could not be loaded.'.format(feeder_id, grid_id))
        try:
            res_grid = pd.concat([res_grid, res_feeder], axis=1, sort=False)
        except:
            print('Feeder {} not added'.format(feeder_id))
    res_grid = res_grid.loc[~res_grid.index.duplicated(keep='last')]
    return res_grid

  # Todo: calculate SOC min
# def create_EV_dataframe (edisgo_obj):
#     em = edisgo_obj.electromobility.charging_processes_df
#     column_names = ['time_index','SOC_min']
#     column_names.extend(em.keys().tolist())
#     EV_df = pd.DataFrame(columns=column_names, index=edisgo_obj.timeseries.timeindex)
#     EV_df["time_index"] = list(range(1,len(edisgo_obj.timeseries.timeindex)+1))
#     print(EV_df)
#     for i in range(1,4): #em.index:
#         for j in range(1,((em["park_end"][i]-em["park_start"][i])+(em["drive_end"][i]-em["drive_start"][i]))):
#             g = j + em["park_start"][i] + em["drive_start"][i]
#             if em["destination"][i] == "driving":                                                                   #timesteps during driving
#                 EV_df["consumption"][g] = em["consumption"][i] / (em["drive_end"][i]-em["drive_start"][i])
#                 EV_df["SOC_min"][g] = 0
#                 EV_df[key][g] = em[key][g] for key in em.keys()
#             elif j == (em["park_end"][i]-em["park_start"][i]) and em["park_end"][i] > 0:                            #last timestep before departure
#                 EV_df[g]["SOC_min"] = min(1, em["consumption"][i+1]/em["bat_cap"][i+1] + 0.2)
#                 EV_df[key][g] = em[key][g] for key in em.keys()
#             elif em["destination"][i] != "driving":
#                 EV_df[g]["SOC_min"] = 0
#                 EV_df[key][g] = em[key][g] for key in em.keys()
#     edisgo_obj[EV_df] = EV_df
#     return edisgo_obj

def create_EV_dict(edisgo_obj, feeder_dir):
    try:
        with open(feeder_dir + r'\{}\{}'.format('electromobility', 'EV_data.pickle'), 'rb') as handle:
            EV_dict = pickle.load(handle)
            handle.close()
    except:
        em = edisgo_obj.electromobility.charging_processes_df
        em[['bat_cap', 'netto_charging_capacity', 'chargingdemand', 'consumption']] = em[['bat_cap', 'netto_charging_capacity', 'chargingdemand', 'consumption']].div(1000)
        timeseries = edisgo_obj.timeseries.charging_points_active_power.iloc[0:672]
        keys = ['car_id', 'car_data', 'charger_id', 'location', 'consumption', 'bat_cap', 'time_series',
                'charger_time_series', 'SOC_min', 'SOC_V2G']
        EV_dict = {key: pd.DataFrame(index=em.car_id.unique(), columns=[1]) for key in keys}
        for key in EV_dict['car_id'].index.tolist():
            EV_dict['bat_cap'][1][key] = em.loc[em['car_id'] == key, ['bat_cap']]['bat_cap'].unique().tolist()
            EV_dict['charger_id'][1][key] = em.loc[em['car_id'] == key, ['charging_park_id', 'charging_point_id']]

        column_names = ['time_index', 'SOC_min']
        column_names.extend(em.keys().tolist())
        EV_dict['time_series'] = {
            key: pd.DataFrame(columns=column_names, index=list(range(1, len(timeseries.index) + 1))) for key in
            em.car_id.unique()}
        EV_dict['car_id'] = em.car_id.unique()

        for i in EV_dict['time_series'].keys(): # Todo: It would be easier to read if you do not use i but time (or whatever i signifies here)
            EV_dict['time_series'][i]['time_index'] = timeseries.index
            EV_dict['time_series'][i]['SOC_min'] = 0

        for i in em.index:
            if em["destination"][i] == "driving":
                EV_dict['time_series'][em['car_id'][i]].loc[
                (em['park_start'][i] + em['drive_start'][i]):(em['park_end'][i] + em['drive_end'][i]),
                em.keys().to_list()] = em.loc[i, :].to_list()
                EV_dict['time_series'][em['car_id'][i]].loc[
                (em['park_start'][i] + em['drive_start'][i]):(em['park_end'][i] + em['drive_end'][i]), 'consumption'] = \
                em["consumption"][i] / (em["drive_end"][i] - em["drive_start"][i] + 1)
                EV_dict['time_series'][em['car_id'][i]].loc[
                (em['park_start'][i] + em['drive_start'][i]):(em['park_end'][i] + em['drive_end'][i]),
                ['charging_park_id', 'charging_point_id']] = 0
            elif em["destination"][i] != "driving":
                EV_dict['time_series'][em['car_id'][i]].loc[
                (em['park_start'][i] + em['drive_start'][i]):(em['park_end'][i] + em['drive_end'][i]),
                em.keys().to_list()] = em.loc[i, :].to_list()
                if em['netto_charging_capacity'][i] == 0 and not em.loc[
                    (em['netto_charging_capacity'] > 0) & (em['destination'] == em['destination'][i]) & (
                            (em['car_id'] == em['car_id'][i]) | (em['use_case'] == 'public') | (
                            em['use_case'] == 'private')), ['netto_charging_capacity', 'charging_park_id']].empty:
                    EV_dict['time_series'][em['car_id'][i]].loc[
                    (em['park_start'][i] + em['drive_start'][i]):(em['park_end'][i] + em['drive_end'][i]),
                    ['netto_charging_capacity', 'charging_park_id']] = \
                        em.loc[(em['netto_charging_capacity'] > 0) & (em['destination'] == em['destination'][i]) & (
                                    (em['car_id'] == em['car_id'][i]) | (em['use_case'] == 'public') | (
                                        em['use_case'] == 'private')), ['netto_charging_capacity',
                                                                        'charging_park_id']].sample().iloc[0,
                        :].tolist()
                if not (i == len(em.index) - 1 or em["car_id"][i] != em["car_id"][i + 1]):
                    EV_dict['time_series'][em['car_id'][i]].loc[em['park_end'][i], 'SOC_min'] = min(1,
                                                                                                    em["consumption"][
                                                                                                        i + 1] /
                                                                                                    em["bat_cap"][
                                                                                                        i + 1] + 0.2)
            if i != len(em.index) - 1 and em['car_id'][i] == em['car_id'][i + 1] and (
                    em['park_end'][i] + em['drive_end'][i] + 1) != (em['park_start'][i + 1] + em['drive_start'][i + 1]):
                print('There is an issue at timestep' + str(i))

        consumption_all = pd.DataFrame(columns=['car_id', 'consumption'])
        SoC_min_all = pd.DataFrame(columns=['car_id', 'SOC_min'])
        charging_points = pd.DataFrame(columns=['car_id', 'charging_park_id'])
        for i in EV_dict['time_series'].keys():
            EV_dict['time_series'][i].set_index('time_index', append=False, inplace=True)
            new_timeline = EV_dict['time_series'][i][['car_id', 'charging_park_id']]
            charging_points = pd.concat([charging_points, new_timeline])
            SoC_min = EV_dict['time_series'][i][['car_id', 'SOC_min']]
            SoC_min_all = pd.concat([SoC_min_all, SoC_min])
            consumption = EV_dict['time_series'][i][['car_id', 'consumption']]
            consumption_all = pd.concat([consumption_all, consumption])

        EV_dict['charger_time_series'] = charging_points.set_index('charging_park_id', append=True).swaplevel(0, -1)
        EV_dict['SOC_min'] = SoC_min_all.set_index('car_id', append=True)
        EV_dict['consumption'] = consumption_all.set_index('car_id', append=True)
        EV_dict['charger_id_set'] = np.sort(charging_points['charging_park_id'].unique()).tolist()
        EV_dict['SOC_V2G'] = 0.7
        EV_dict['P_max_ev'] = 0.135
        EV_dict['charging_efficiency'] = 0.9
        EV_dict['Car_Characteristics'] = pd.DataFrame(
            index=['BEV Kleinwagen', 'BEV Mittelklasse', 'BEV Oberklasse', 'PHEV Kleinwagen', 'PHEV Mittelkasse',
                   'PHEV Oberklasse'], columns=['Maximale Ladeleistung', 'Nutzbare Batteriekapazitaet'])
        EV_dict['Car_Characteristics']['Maximale Ladeleistung'] = [0.120, 0.350, 0.350, 0.120, 0.120, 0.120]
        EV_dict['Car_Characteristics']['Nutzbare Batteriekapazitaet'] = [0.030, 0.065, 0.090, 0.020, 0.020, 0.020] #Todo: Please use english instead :)

        EV_dict['charger_time_series'] = {key: pd.DataFrame() for key in EV_dict['charger_id_set']}
        for i in EV_dict['charger_id_set']:
            a = charging_points.set_index('charging_park_id', append=True).swaplevel(0, -1).loc[i]
            a.reset_index(inplace=True)
            gb = a.groupby(['index'])
            result = gb['car_id'].unique()
            EV_dict['charger_time_series'][i] = result
        with open(feeder_dir + r'\{}\{}'.format('electromobility', 'EV_data.pickle'), 'wb') as handle:
            pickle.dump(EV_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

    edisgo_obj.electromobility.EV_dict = EV_dict
    return edisgo_obj

    # keys = ['car_id', 'car_data', 'charger_id', 'location', 'consumption', 'bat_cap', 'time_series', 'SOC_min']
    # em = edisgo_obj.electromobility.charging_processes_df
    # timeseries = edisgo_obj.timeseries.charging_points_active_power
    # EV_dict = {key: pd.DataFrame(index=em.car_id.unique(), columns=[1]) for key in keys}
    # for key in EV_dict['car_id'].index.tolist():
    #     EV_dict['bat_cap'][1][key] = em.loc[em['car_id'] == key, ['bat_cap']]['bat_cap'].unique().tolist()
    #     EV_dict['charger_id'][1][key] = em.loc[em['car_id'] == key, ['charging_park_id', 'charging_point_id']]
    #
    # column_names = ['time_index', 'SOC_min']
    # column_names.extend(em.keys().tolist())
    # EV_dict['time_series'] = {key: pd.DataFrame(columns=column_names, index=timeseries.index) for key in
    #                           em.car_id.unique()}
    #
    # for i in EV_dict['time_series'].keys():
    #     EV_dict['time_series'][i]['time_index'] = list(range(1, len(timeseries.index) + 1))
    #
    # for i in em.index:
    #     for j in range(0,
    #                    ((em["park_end"][i] - em["park_start"][i]) + (em["drive_end"][i] - em["drive_start"][i]) + 1)):
    #         g = j + em["park_start"][i] + em["drive_start"][i] - 1
    #         if em["destination"][i] == "driving":  # timesteps during driving
    #             EV_dict['time_series'][em['car_id'][i]]["SOC_min"][g] = 0
    #             for key in em.keys():
    #                 EV_dict['time_series'][em['car_id'][i]][key][g] = em[key][i]
    #             EV_dict['time_series'][em['car_id'][i]]["consumption"][g] = em["consumption"][i] / (
    #                         em["drive_end"][i] - em["drive_start"][i] + 1)
    #             print('Loop driving, g: ' + str(g) + ' j: ' + str(j) + ' i: ' + str(i))
    #         elif (j == (em["park_end"][i] - em["park_start"][i])) and (
    #                 em["park_end"][i] > 0):  # last timestep before departure
    #             if i == len(em.index) - 1 or em["car_id"][i] != em["car_id"][i + 1]:
    #                 EV_dict['time_series'][em['car_id'][i]]["SOC_min"][g] = 0
    #             else:
    #                 EV_dict['time_series'][em['car_id'][i]]["SOC_min"][g] = min(1, em["consumption"][i + 1] /
    #                                                                             em["bat_cap"][i + 1] + 0.2)
    #             for key in em.keys():
    #                 EV_dict['time_series'][em['car_id'][i]][key][g] = em[key][i]
    #             print('Loop park_end, g: ' + str(g) + ' j: ' + str(j) + ' i: ' + str(i))
    #         elif em["destination"][i] != "driving":
    #             EV_dict['time_series'][em['car_id'][i]]["SOC_min"][g] = 0
    #             for key in em.keys():
    #                 EV_dict['time_series'][em['car_id'][i]][key][g] = em[key][i]
    #             print('Loop else, g: ' + str(g) + ' j: ' + str(j) + ' i: ' + str(i))
    # edisgo_obj.electromobility.EV_dict = EV_dict
    # return edisgo_obj





    # keys = ['car_id', 'car_data', 'charger_id', 'location', 'consumption', 'bat_cap']
    # EV_dict = {key: pd.DataFrame(index=edisgo_obj.electromobility.charging_processes_df.car_id.unique()) for key in keys}
    # EV_dict['car_id'] = edisgo_obj.electromobility.charging_processes_df.car_id.unique()
    # EV_dict['car_data'] = pd.DataFrame(index=)
    # EV_dict['charger_id'] =
