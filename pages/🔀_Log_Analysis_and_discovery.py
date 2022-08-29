
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pm4py as pm
import os
from PIL import Image
import plotly.express as pe
import missingno as msn
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.visualization.graphs import visualizer as graphs_visualizer
from pyvis.network import Network
from pm4py.visualization.sna import visualizer as sna_visualizer
import streamlit.components.v1 as components
import graphviz
import pydotplus
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer
import shutil
import tempfile
from enum import Enum
from pm4py.util import exec_utils, vis_utils

#Page Layout and config

st.set_page_config(
     page_title="Log Analysis and Process Discovery",
     page_icon=":clipboard:",
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

    st.title("Show me a Path")

    log_file_input = st.file_uploader("Upload the event log file", accept_multiple_files=False, type = [".xlsx",".csv",".xes"], help = "The event log should have the following structure: ID, Timestamp, Activity, Resource")

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
        
        tab1,tab2,tab3,tab4,tab5 = st.tabs([" Event log information","Time and Performance Analysis","Activities Report","Process Discovery","Social Network"])
        # Tab 1 - Log File Info
        with tab1:
            st.header(":page_facing_up:" "Log File and structure")
            st.subheader("Basic Information")
            st.write("There are **{} patients** in this event log".format(log["case:concept:name"].nunique()))
            st.dataframe(log)
            st.subheader("Missing Data")
            md = msn.matrix(log)
            st.pyplot(md.figure)
        # Tab 2 - Time and Performance Analysis
        with tab2:
            #Gráfico 1 - Case Durations
            st.header(":clock2:" "Time and Performance Analysis")
            st.subheader("Duration of Traces")
            all_case_durations = pm.get_all_case_durations(elog)
            durations = pd.DataFrame(all_case_durations,columns=["durat"])
            durations.durat = durations.durat.apply(lambda x : (((x/60)/60)/24))
            fig5=pe.histogram(durations,
            template="simple_white",
            width = 700,
            height = 400,
            labels={"value":"Nº de dias"},
            marginal="box")
            fig5.update_layout(showlegend=False)
            st.plotly_chart(fig5)
            #Gráfico 2 - Log density over time
            st.subheader("Log Density over time")
            x, y = attributes_filter.get_kde_date_attribute(elog, attribute="time:timestamp")
            gviz = graphs_visualizer.apply_plot(x, y, variant=graphs_visualizer.Variants.DATES)
            image = Image.open(gviz)
            st.image(image)
            # Gráficos 3 - Days of the week
            st.subheader("Distribution of log by day of the week")
        # Tab 3 - Activities Report
        with tab3:
            st.header(":bar_chart:" "Activities Report")
            #Gráfico 1 - N activities
            st.subheader("Number of activities by patient")
            n_activities = log.groupby("case:concept:name")["concept:name"].count()
            n_act_graph = pe.box(n_activities,
            width = 700,
            height = 400,
            template = "simple_white",
            orientation="h",
            labels = {"value" : "N Activities", "variable" : ""})
            st.plotly_chart(n_act_graph)
            #Gráfico 2 - Most common activities
            st.subheader("Top10 activities")
            log3=log[["case:concept:name","concept:name"]]
            gb=log3.value_counts().to_frame("counts").reset_index()
            d_act=gb["concept:name"].value_counts().head(10).to_frame()
            d_act["concept:name"]=d_act["concept:name"].apply(lambda x : (x*100)/log["case:concept:name"].nunique())
            d_act["concept:name"]=d_act["concept:name"].round(decimals=1)
            fig2=pe.bar(d_act,
            template="simple_white",
            width=700,
            height=400,
            labels={"value":"%","index":""},
            color_discrete_sequence=["Teal"])
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2)
            #Gráfico 3 - Traces
            st.subheader("Most common traces")
            list_cases = log["case:concept:name"].unique()
            Traces = []
            ids = []
            for i in list_cases:
                a = log.loc[log["case:concept:name"] == "{}".format(i)].sort_values(["time:timestamp"])
                b = a["concept:name"].str.cat(sep=",")
                ids.append(i)
                Traces.append(b)
            trace_db = pd.DataFrame({"ID":ids,"Trace":Traces})
            trace_db = trace_db.Trace.value_counts().head(10).to_frame().reset_index()
            trace_db.Trace = trace_db.Trace.apply(lambda x : (x*100)/log["case:concept:name"].nunique())
            trace_db.Trace = trace_db.Trace.round(decimals=1)
            fig4 = pe.bar(trace_db, x = "Trace", y = "index",
            orientation="h",
            width = 1000,
            height = 400,
            template="simple_white",
            labels={"index":"","Trace":"%"},
            color_discrete_sequence=["Teal"],
            text="Trace").update_yaxes(categoryorder="total ascending").update_xaxes(visible=False)
            fig4.update_layout(yaxis=dict(tickfont=dict(size=8)))
            fig4.update_traces(textposition="outside")
            st.plotly_chart(fig4)

            
        with tab4:
            filtered_log = pm.filter_case_size(elog, 3, 30)
            heu_net = pm.discover_heuristics_net(filtered_log, dependency_threshold=0.99, loop_two_threshold=0.99)
            path = os.getcwd()
            graph = pm.save_vis_heuristics_net(heu_net,file_path = f"{path}/heunet.png")
            image = Image.open("heunet.png")
            st.image(image)
        #Tab 5 - Social Network
        with tab5:
            ### THIS SECTION USES ORIGINAL PM4PY VIS SCRIPT TO ALLOW CUSTOMIZING PYVIS NETWORK
            ## START

            class Parameters(Enum):
                WEIGHT_THRESHOLD = "weight_threshold"


            def get_temp_file_name(format):
                """
                Gets a temporary file name for the image

                Parameters
                ------------
                format
                    Format of the target image
                """
                filename = tempfile.NamedTemporaryFile(suffix='.' + format)

                return filename.name


            def apply(metric_values, parameters=None):
                """
                Perform SNA visualization starting from the Matrix Container object
                and the Resource-Resource matrix

                Parameters
                -------------
                metric_values
                    Value of the metrics
                parameters
                    Possible parameters of the algorithm, including:
                        - Parameters.WEIGHT_THRESHOLD -> the weight threshold to use in displaying the graph

                Returns
                -------------
                temp_file_name
                    Name of a temporary file where the visualization is placed
                """
                from pyvis.network import Network

                if parameters is None:
                    parameters = {}

                weight_threshold = exec_utils.get_param_value(Parameters.WEIGHT_THRESHOLD, parameters, 0)
                directed = metric_values[2]

                temp_file_name = get_temp_file_name("html")

                rows, cols = np.where(metric_values[0] > weight_threshold)
                weights = list()

                for x in range(len(rows)):
                    weights.append(metric_values[0][rows[x]][cols[x]])

                got_net = Network(height="750px", width="100%", bgcolor="white", font_color="#f5b642", directed=directed)
                # set the physics layout of the network
                got_net.barnes_hut()

                edge_data = zip(rows, cols, weights)

                for e in edge_data:
                    src = metric_values[1][e[0]]  # convert ids to labels
                    dst = metric_values[1][e[1]]
                    w = e[2]

                # I have to add some options here, there is no parameter
                highlight = {'border': "#164bdb", 'background': "#16b7db"}
                # color = {'border': "#000000", 'background': "#123456"}
                got_net.add_node(src, src, title=src, labelHighlightBold=True, color={'highlight': highlight})
                got_net.add_node(dst, dst, title=dst, labelHighlightBold=True, color={'highlight': highlight})
                got_net.add_edge(src, dst, value=w, title=w)

                neighbor_map = got_net.get_adj_list()

                dict = got_net.get_edges()

                # add neighbor data to node hover data
                for node in got_net.nodes:
                    counter = 0
                    if directed:
                        node["title"] = "<h3>" + node["title"] + " Output Links: </h3>"
                    else:
                        node["title"] = "<h3>" + node["title"] + " Links: </h3>"
                    for neighbor in neighbor_map[node["id"]]:
                        if (counter % 10 == 0):
                            node["title"] += "<br>::: " + neighbor
                        else:
                            node["title"] += " ::: " + neighbor
                        node["value"] = len(neighbor_map[node["id"]])
                        counter += 1

                got_net.set_options("""
                var options = {
                    "nodes": {
                        "borderwidth" : 1,
                        "borderwidthselected" : 2,
                        "color": {
                            "background" : "rgba(114,191,197,1)",
                            "border" : "rgba(0,100,121,1)",
                            "highlight" : {
                                "color" : "rgba(22,78,219,1)"
                            }
                        },
                        "font": {
                            "size": 70,
                            "color" : "rgba(219,111,22,1)",
                            "face" : "verdana"
                        },
                        "labelHighlightBold": true,
                        "physics": true,
                        "scaling": {
                            "min": 10,
                            "max": 30
                        },
                        "shape" : "ellipse",
                        "shapeProperties": {
                            "borderRadius": 6,
                            "interpolation": true
                        }
                    },
                    "edges": {
                        "arrowStrikethrough": true,
                        "color": {
                            "color" : "rgba(114,191,197,1)",
                            "inherit": true
                        },
                        "hoverWidth" : 1.5,
                        "labelHighlightBold" : true,
                        "physics" : true,
                        "scaling": {
                            "min": 1,
                            "max": 15
                        },
                        "selectionWidth" : 1.5,
                        "selfReferenceSize" : 20,
                        "selfReference": {
                            "size" : 20,
                            "angle": 0.7853981633974483,
                            "renderBehindTheNode" : true
                        },
                        "smooth": {
                            "type" : "dynamic",
                            "forceDirection": "none",
                            "roundness" : 0.5
                        },
                        "width" : 1
                    },
                    "physics": {
                        "barnesHut": {
                            "gravitationalConstant": -80000,
                            "springLenght" : 250,
                            "springConstant" : 0.001
                        },
                    "minVelocity": 0.75
                    }
                }
                """)
                

                got_net.write_html(temp_file_name)
                st.write([i for i in edge_data])


                return temp_file_name
            
            def save(temp_file_name, dest_file, parameters=None):

                if parameters is None:
                    parameters = {}

                shutil.copyfile(temp_file_name, dest_file)

            st.subheader(":male-doctor:""Social Network")
            hw_values = pm.discover_handover_of_work_network(elog)
            gviz2 = apply(hw_values)
            path = os.getcwd()
            save(gviz2,f"{path}/sna.html")
            HtmlFile = open(f"{path}/sna.html")
            components.html(HtmlFile.read(), width = 800, height = 800)
    

