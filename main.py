import pandas as pd
import pm4py
import os

from pm4py.algo.conformance.tokenreplay import algorithm as token_based_replay

import activities_list
import conformance_check

if __name__ == '__main__':

    # importa il dataset e divide il log in Outdoor e Indoor
    dataset_csv = pd.read_csv("data/training_data.CSV", sep=';')

    # crea dataframe
    dataset_df = pm4py.format_dataframe(dataset_csv, case_id='identifier', activity_key='activity', timestamp_key='timestamp')
    log1 = pm4py.convert_to_event_log(dataset_df)

    # crea due log, uno per le attività indoor e uno per quelle outdoor
    in_log = pm4py.filter_event_attribute_values(log1, 'activity', activities_list.activities_indoor, level="event", retain=True)
    out_log = pm4py.filter_event_attribute_values(log1, 'activity', activities_list.activities_outdoor, level="event", retain=True)

    # ALPHA MINER
    alpha_in_net, alpha_in_im, alpha_in_fm = pm4py.discover_petri_net_alpha(in_log)
    alpha_out_net, alpha_out_im, alpha_out_fm = pm4py.discover_petri_net_alpha(out_log)

    # INDUCTIVE MINER
    ind_in_net, ind_in_im, ind_in_fm = pm4py.discover_petri_net_inductive(in_log)
    ind_out_net, ind_out_im, ind_out_fm = pm4py.discover_petri_net_inductive(out_log)

    # Heuristic Net
    indoor_map = pm4py.discover_heuristics_net(in_log)
    outdoor_map = pm4py.discover_heuristics_net(out_log)

    pm4py.write_pnml(alpha_in_net, alpha_in_im, alpha_in_fm, "petri nets/alpha_indoor_petrinet.pnml")
    pm4py.write_pnml(ind_in_net, ind_in_im, ind_in_fm, "petri nets/inductive_indoor_petrinet.pnml")

    pm4py.write_pnml(alpha_out_net, alpha_out_im, alpha_out_fm, "petri nets/alpha_outdoor_petrinet.pnml")
    pm4py.write_pnml(ind_out_net, ind_out_im, ind_out_fm, "petri nets/inductive_outdoor_petrinet.pnml")


    # CONFORMANCE CHECKING con Token-Based Replay
    csv_log = pd.read_csv("data/test_data.CSV", sep=';')
    df_pr = pm4py.format_dataframe(csv_log, case_id='identifier', activity_key='activity', timestamp_key='timestamp')

    test_log = pm4py.convert_to_event_log(df_pr)

    indoor_test_log = pm4py.filter_event_attribute_values(test_log, 'activity', activities_list.activities_indoor, level="event", retain=True)
    outdoor_test_log = pm4py.filter_event_attribute_values(test_log, 'activity', activities_list.activities_outdoor, level="event", retain=True)


    print("\n ALPHA MINER")
    print("\nIndoor activities")
    conformance_check.conformance_checking(indoor_test_log, alpha_in_net, alpha_in_im, alpha_in_fm)

    print("\nOutdoor activities")
    conformance_check.conformance_checking(outdoor_test_log, alpha_out_net, alpha_out_im, alpha_out_fm)


    print("\n INDUCTIVE MINER")
    print("\nIndoor activities")
    conformance_check.conformance_checking(indoor_test_log, ind_in_net, ind_in_im, ind_in_fm)

    print("\nOutdoor activities")
    conformance_check.conformance_checking(outdoor_test_log, ind_out_net, ind_out_im, ind_out_fm)


