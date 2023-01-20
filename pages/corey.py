import streamlit as st
import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

def corey(s, krw_sor, nw):
    se = (s - np.min(s)) / (1 - np.min(s) - (1 - np.max(s)))
    krw = krw_sor * se ** nw
    return krw

def data_clean(file_name):
    df = pd.read_excel(file_name, header=0, skiprows=[1])
    df = df[["ID PK", "Sw", "Krw"]]
    # remove the rows where krw is above 1
    df = df[df["Krw"] <= 1.0]
    df = df.dropna()
    grouped_data = df.groupby("ID PK")[["Sw", "Krw"]]
    return grouped_data

def get_corey_params(grouped_data):
    corey_params = {}
    #  (param1min, param2min), (param1max, param2max)
    param_bounds_corey = [(0, 1), (1, 8)]
    for i, (well_id, well_data) in enumerate(grouped_data):
        #if i > 10:
        #    break
        Sw = np.array(well_data["Sw"]) / 100
        krw = np.array(well_data["Krw"])
        # print(well_id)
        # print(Sw)
        # print(krw)
        popt, _ = curve_fit(corey, Sw, krw, bounds=param_bounds_corey)
        krw_sor, nw = popt
        corey_params[well_id] = (krw_sor, nw)
    return corey_params

def plot_corey_curve_params(corey_params, grouped_data):
    for well_id, (krw_sor, nw) in corey_params.items():
        well_data = grouped_data.get_group(well_id)
        Sw = np.array(well_data["Sw"]) / 100
        krw = np.array(well_data["Krw"])
        fig, ax = plt.subplots()
        ax.scatter(Sw, krw, label='Data')
        # Generate points on the Corey curve
        s_values = np.linspace(np.min(Sw), np.max(Sw), 100)
        krw_model = corey(s_values, krw_sor, nw)
        ax.plot(s_values, krw_model, label='Corey Model')
        ax.set_xlabel('Sw')
        ax.set_ylabel('krw')
        ax.legend()
        ax.set_title(f'well_id: {well_id}')
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        st.pyplot(fig)

@st.cache
def convert_results(corey_params):
    # Convert the dictionary to a DataFrame
    corey_params_df = pd.DataFrame.from_dict(corey_params, orient='index', columns=["krw_sor", "nw"])
    # Save the DataFrame to a CSV file
    return corey_params_df.to_csv().encode('utf-8')


file = st.file_uploader("Upload Excel file", type="xlsx")
if file is not None:
    file_name = file.name
    grouped_data = data_clean(file_name)
    corey_params = get_corey_params(grouped_data)
    csv = convert_results(corey_params)
    st.download_button(
        label="Download parameters as CSV",
        data=csv,
        file_name='corey_params_water.csv',
        mime='text/csv',
    )
    plot_corey_curve_params(corey_params, grouped_data)