import pm4py

def conformance_checking(log, net, im, fm):

    if len(log) == 0:
        print(" log of indoor activites is empty")
    else:
        replayed_traces = pm4py.conformance_diagnostics_token_based_replay(log, net, im, fm)
        i=0
        print(" ")
        while i < len(replayed_traces):
            print("Processo num. ", i+1)
            if( (replayed_traces[i]['trace_is_fit']) == True ):
                print("\t Regular! Fitness:",  round(replayed_traces[i]['trace_fitness'],2))
            else:
                print("\t Not regular. Fitness:",  round(replayed_traces[i]['trace_fitness'],2))
            i=i+1