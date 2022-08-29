
import streamlit as st
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

st.set_page_config(layout="wide")

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
    st.header("Home Page")
