
st.header(":page_facing_up:" "Log File and structure")
    st.subheader("Basic Information")
    st.write("There are **{} patients** in this event log".format(log["case:concept:name"].nunique()))
    st.dataframe(log)
