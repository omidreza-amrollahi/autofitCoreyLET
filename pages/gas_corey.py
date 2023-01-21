import streamlit as st
import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

def corey(s, krg_swc, ng):
    se = (s - np.min(s)) / (1 - np.min(s) - (1 - np.max(s)))
    krg = krg_swc * np.abs(1 - se) ** ng
    return krg

def data_clean(file):
    df = pd.read_excel(file, header=0, skiprows=[1])
    df = df[["ID PK", "Sw", "Krg"]]
    # remove the rows where krg is above 1
    df = df[df["Krg"] <= 1.0]
    df = df.dropna()
    grouped_data = df.groupby("ID PK")[["Sw", "Krg"]]
    return grouped_data

def get_corey_params(grouped_data, n_lb=8, n_ub=1):
    corey_params = {}
    #  (param1min, param2min), (param1max, param2max)
    param_bounds_corey = [(0, n_lb), (1, n_ub)]
    for i, (well_id, well_data) in enumerate(grouped_data):
        #if i > 10:
        #    break
        Sw = np.array(well_data["Sw"]) / 100
        krg = np.array(well_data["Krg"])
        # print(well_id)
        # print(Sw)
        # print(krg)
        popt, _ = curve_fit(corey, Sw, krg, bounds=param_bounds_corey)
        krg_swc, ng = popt
        corey_params[well_id] = (krg_swc, ng)
    return corey_params

def plot_corey_curve_params(corey_params, grouped_data):
    for well_id, (krg_swc, ng) in corey_params.items():
        well_data = grouped_data.get_group(well_id)
        Sw = np.array(well_data["Sw"]) / 100
        krg = np.array(well_data["Krg"])
        fig, ax = plt.subplots()
        ax.scatter(Sw, krg, label='Data')
        # Generate points on the Corey curve
        s_values = np.linspace(np.min(Sw), np.max(Sw), 100)
        krg_model = corey(s_values, krg_swc, ng)
        ax.plot(s_values, krg_model, label='Corey Model')
        ax.set_xlabel('Sw')
        ax.set_ylabel('krg')
        ax.legend()
        ax.set_title(f'well_id: {well_id}')
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        st.pyplot(fig)

@st.cache
def convert_results(corey_params):
    # Convert the dictionary to a DataFrame
    corey_params_df = pd.DataFrame.from_dict(corey_params, orient='index', columns=["krg_swc", "ng"])
    # Save the DataFrame to a CSV file
    return corey_params_df.to_csv().encode('utf-8')

st.title("Gas Corey fitting")
c1, c2 = st.columns(2)
with st.container():
    exponent_lb = c1.number_input("Corey exponent lower boundary:", 1, 20, 1)
    exponent_ub = c2.number_input("Corey exponent upper boundary:", 1, 20, 8)
    

file = st.file_uploader("Upload Excel file", type="xlsx")
if file is not None:
    grouped_data = data_clean(file)
    corey_params = get_corey_params(grouped_data, exponent_lb, exponent_ub)
    csv = convert_results(corey_params)
    st.download_button(
        label="Download parameters as CSV",
        data=csv,
        file_name='corey_params_gas.csv',
        mime='text/csv',
    )
    plot_corey_curve_params(corey_params, grouped_data)