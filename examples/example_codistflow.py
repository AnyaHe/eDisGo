import copy
import os

import matplotlib.pyplot as plt
import pandas as pd

import edisgo.opf.lopf as opt
from edisgo.edisgo import import_edisgo_from_files
from edisgo.network.electromobility import (Electromobility,
                                            get_energy_bands_for_optimization)
from edisgo.network.timeseries import get_component_timeseries
from Script_prepare_grids_for_optimization import \
    get_downstream_nodes_matrix_iterative

grid_dir = "minimum_working"
opt_ev = True
opt_stor = False
save_res = False

if os.path.isfile("x_charge_ev_pre.csv"):
    ts_pre = pd.read_csv("x_charge_ev_pre.csv", index_col=0, parse_dates=True)
else:
    ts_pre = pd.DataFrame()

timeindex = pd.date_range("2011-01-01", periods=8760, freq="h")
storage_ts = pd.DataFrame({"Storage 1": 8760 * [0]}, index=timeindex)

edisgo = import_edisgo_from_files(grid_dir)
get_component_timeseries(
    edisgo,
    timeseries_load="demandlib",
    timeseries_generation_fluctuating="oedb",
    timeseries_storage_units=storage_ts,
)
timesteps = edisgo.timeseries.timeindex[7 * 24 : 2 * 24 * 7]

cp_id = 1
ev_data = pd.read_csv(
    os.path.join(grid_dir, "BEV_standing_times_minimum_working.csv"), index_col=0
)
charging_events = ev_data.loc[ev_data.chargingdemand > 0]
charging_events["charging_park_id"] = cp_id
Electromobility(edisgo_obj=edisgo)
edisgo.electromobility.charging_processes_df = charging_events
cp = edisgo.add_component(
    "ChargingPoint", bus="Bus 2", p_nom=0.011, use_case="home", add_ts=False
)
ts_cp = pd.DataFrame(columns=[cp], index=edisgo.timeseries.timeindex, data=0)
edisgo.timeseries.charging_points_active_power = ts_cp
edisgo.timeseries.charging_points_reactive_power = ts_cp
edisgo.electromobility.integrated_charging_parks_df = pd.DataFrame(
    index=[cp_id], columns=["edisgo_id"], data=cp
)
edisgo.electromobility.simbev_config_df = pd.DataFrame(
    index=["eta_CP", "stepsize"], columns=["value"], data=[0.9, 60]
)
energy_bands = get_energy_bands_for_optimization(edisgo_obj=edisgo, use_case="home")

downstream_node_matrix = get_downstream_nodes_matrix_iterative(edisgo.topology)
parameters = opt.prepare_time_invariant_parameters(
    edisgo,
    downstream_node_matrix,
    pu=False,
    optimize_storage=False,
    optimize_ev_charging=True,
    ev_flex_bands=energy_bands,
)
model = opt.setup_model(
    parameters,
    timesteps=timesteps,
    objective="residual_load",
    optimize_storage=False,
    v_min=1.0,
)
losses = pd.DataFrame(index=timesteps, columns=model.branches.index, data=1)
losses_pre = pd.DataFrame(index=timesteps, columns=model.branches.index, data=0)
tol = 1e-9
max_iter = 100

for iter in range(max_iter):
    if ((losses - losses_pre).abs() > tol).any().any():
        print(
            f"Losses: Starting iteration {iter}. \n"
            f"Maximum absolute deviation: {(losses-losses_pre).abs().max().max()}"
        )
        results = opt.optimize(model, "gurobi")

        edisgo_obj = copy.deepcopy(model.edisgo_obj)
        # iteration losses
        losses_pre = (
            pd.Series(model.losses.extract_values()).unstack().T.set_index(timesteps)
        )
        # Add optimised ev charging
        edisgo_obj.timeseries._charging_points_active_power.loc[
            :, results["x_charge_ev"].columns
        ] = results["x_charge_ev"]
        # Add curtailment as new loads and feedin to grid
        curtailment_load = results["curtailment_load"] + results["curtailment_ev"]
        curtailment_load = curtailment_load[
            curtailment_load.columns[curtailment_load.sum() > 0]
        ]
        curtailment_reactive_load = results["curtailment_reactive_load"][
            curtailment_load.columns
        ]
        curtailment_feedin = results["curtailment_feedin"]
        curtailment_feedin = curtailment_feedin[
            curtailment_feedin.columns[curtailment_feedin.sum() > 0]
        ]
        curtailment_reactive_feedin = results["curtailment_reactive_feedin"][
            curtailment_feedin.columns
        ]

        edisgo_obj.timeseries.mode = "manual"
        edisgo_obj.timeseries.timeindex = timesteps
        edisgo_obj.add_components(
            "Load",
            ts_active_power=curtailment_feedin,
            ts_reactive_power=curtailment_reactive_feedin,
            buses=curtailment_feedin.columns,
            load_ids=curtailment_feedin.columns,
            peak_loads=curtailment_feedin.max().values,
            annual_consumptions=curtailment_feedin.sum().values,
            sectors=["feedin_curtailment"] * len(curtailment_feedin.columns),
        )
        print(
            "Load added for curtailment at buses {}".format(curtailment_feedin.columns)
        )
        edisgo_obj.add_components(
            "Generator",
            ts_active_power=curtailment_load,
            ts_reactive_power=curtailment_reactive_load,
            buses=curtailment_load.columns,
            generator_ids=curtailment_load.columns,
            p_noms=curtailment_load.max().values,
            generator_types=["load_curtailment"] * len(curtailment_load.columns),
        )
        print(
            "Generator added for curtailment at buses {}".format(
                curtailment_load.columns
            )
        )
        # run power flow and extract losses
        pypsa_obj = edisgo_obj.to_pypsa()
        pypsa_obj.pf(timesteps)
        losses = pd.concat(
            [
                pypsa_obj.lines_t.p0 + pypsa_obj.lines_t.p1,
                pypsa_obj.transformers_t.p0 + pypsa_obj.transformers_t.p1,
            ],
            axis=1,
        )
        losses_q = pd.concat(
            [
                pypsa_obj.lines_t.q0 + pypsa_obj.lines_t.q1,
                pypsa_obj.transformers_t.q0 + pypsa_obj.transformers_t.q1,
            ],
            axis=1,
        )
        # update model with losses
        opt.update_losses(model, losses)
    else:
        break

results["x_charge_ev"].plot()
plt.show()
if not ts_pre.empty:
    ts_pre.plot()
    plt.show()
    pd.testing.assert_frame_equal(ts_pre, results["x_charge_ev"])
if save_res:
    results["x_charge_ev"].to_csv("x_charge_ev_pre.csv")
print("SUCCESS")
