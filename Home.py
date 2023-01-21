import streamlit as st


st.title("Relative permeability autofit")
st.write("Use the tab menu on the left to navigate to the different pages.")
st.write("Guideline: The input data file needs to be an excel file with the well IDs in a column named 'ID PK', the water saturation in a column named 'Sw', the relative permeability in a column named 'Krw', and gas relative permeability in a column named 'Krg'.")
st.write("Units: Water saturation are input in percentage, and relative permeabilities are given in fraction.")
st.write("Aknowledgement: This app is developed by [Department of Petroleum Engineering Leoben](dpe.at) in collaboration with [OMV AG.](https://www.omv.com/en).")
st.write("Contact: [Omidreza Amrollahinasab](https://www.linkedin.com/in/amrollahinasab/)")