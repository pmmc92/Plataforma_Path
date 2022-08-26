
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pm4py as pm
import os
import plotly.express as pe

st.set_page_config(
     page_title="Conformance Analysis",
     page_icon=":heavy_check_mark:",
     layout="wide")

# Password check

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

if check_password() == True:


    # App structure

    st.title("Conformance Analysis")
    
    #Upload Log File

    log_file_input = st.file_uploader("Upload the event log file", accept_multiple_files=False, type = [".xlsx",".csv",".xes"], help = "The event log should have the following structure: ID, Timestamp, Activity, Resource")

    # Upload BPMN
    bpmn_file_input = st.file_uploader("Upload the BPMN file", accept_multiple_files=False, type = [".bpmn"])

    if st.button("Analisar"):
        if (os.path.splitext(log_file_input.name)[1]) == ".csv":
            log = pd.read_csv(log_file_input)
            log = log.rename(columns={log.columns[3]:"org:resource"})
            log = pm.format_dataframe(log, case_id = log.columns[0], activity_key=log.columns[2], timestamp_key=log.columns[1])
            log = log[['org:resource', 'case:concept:name', 'concept:name', 'time:timestamp']]
            log["org:resource"] = log["org:resource"].astype(str)
            elog = pm.convert_to_event_log(log)
        elif (os.path.splitext(log_file_input.name)[1]) == ".xlsx":
            log = pd.read_excel(log_file_input)
            log = log.rename(columns={log.columns[3]:"org:resource"})
            log = pm.format_dataframe(log, case_id = log.columns[0], activity_key=log.columns[2], timestamp_key=log.columns[1])
            log = log[['org:resource', 'case:concept:name', 'concept:name', 'time:timestamp']]
            log["org:resource"] = log["org:resource"].astype(str)
            elog = pm.convert_to_event_log(log)
        else:
            log = pm.read_xes(log_file_input)
            filtered_log = pm.filter_case_size(log, 3, 30)
            heu_net = pm.discover_heuristics_net(filtered_log, dependency_threshold=0.99, loop_two_threshold=0.99)
            pm.save_vis_heuristics_net(heu_net, "net.png")
            st.image("net.png")
        # Read BPMN and transform to petrinet
        bpmn_graph = pm.read_bpmn(bpmn_file_input)
        net, im, fm = pm.convert_to_petri_net(bpmn_graph)
        tab1,tab2 = st.tabs(["Token Replay Method","Alignments Method"])
        # Token Replay Method
        with tab1:
            replayed_traces = pm.conformance_diagnostics_token_based_replay(elog, net, im, fm)
            list_tf = []
            for i in range(len(replayed_traces)):
                trace_sel = replayed_traces[i]
                tf_t = trace_sel.get("trace_fitness")
                list_tf.append(tf_t)
            st.header("Token Replay Method")
            st.markdown("Token-based replay technique is a conformance checking algorithm that checks how well a process conforms with its model by replaying each trace on the model.Using the four counters produced tokens, consumed tokens, missing tokens, and remaining tokens, it records the situations where a transition is forced to fire and the remaining tokens after the replay ends. Based on the count at each counter, we can compute the fitness value between the trace and the model.")
            st.subheader("The trace fitness value is **{:.2f}**".format(np.mean(list_tf)))
            tf_g1=pe.histogram(list_tf,
            template="simple_white",
            width = 900,
            height = 400,
            labels={"variable":"Trace Fitness"},
            color_discrete_sequence=["teal"],
            marginal="box").update_xaxes(visible=False)
            tf_g1.update_layout(showlegend=False)
            st.plotly_chart(tf_g1)
        #Alignment Method
        with tab2:
            aligned_traces = pm.conformance_diagnostics_alignments(elog,net,im,fm)
            list_tf2 = []
            for i in range(len(aligned_traces)):
                trace_sel2 = aligned_traces[i]
                tf_t2 = trace_sel2.get("fitness")
                list_tf2.append(tf_t2)
            st.header("Alignment method")
            st.markdown("Alignments is a technique, which performs an exhaustive search to find out the optimal alignment between the observed trace and the process model. Hence, it is guaranteed to return the closest model run in comparison to the trace.")
            st.subheader("The trace fitness value is **{:.2f}**".format(np.mean(list_tf2)))
            tf_g2=pe.histogram(list_tf2,
            template="simple_white",
            width = 900,
            height = 400,
            labels={"variable":"Trace Fitness"},
            color_discrete_sequence=["teal"],
            marginal="box").update_xaxes(visible=False)
            tf_g2.update_layout(showlegend=False)
            st.plotly_chart(tf_g2)

    
