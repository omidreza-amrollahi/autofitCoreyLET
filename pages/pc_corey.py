import streamlit as st
import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

def corey(s, pd, pc_lambda):
    se = (s - np.min(s)) / (1 - np.min(s) - (1 - np.max(s)))
    se[0] = 1e-3
    pc = pd * (se) ** (-1 / pc_lambda)
    return pc

def data_clean(file):
    df = pd.read_excel(file, header=0, skiprows=[1])
    df = df[["ID PK", "Sw", "Pc"]]
    df = df.dropna()
    grouped_data = df.groupby("ID PK")[["Sw", "Pc"]]
    return grouped_data

def get_corey_params(grouped_data, pd_lb, pd_ub, lambda_lb=0.5, lambda_ub=2):
    corey_params = {}
    #  (param1min, param2min), (param1max, param2max)
    param_bounds_corey = [(pd_lb, lambda_lb), (pd_ub, lambda_ub)]
    for i, (well_id, well_data) in enumerate(grouped_data):
        #if i > 10:
        #    break
        Sw = np.array(well_data["Sw"]) / 100
        pc = np.array(well_data["Pc"])
        # print(well_id)
        # print(Sw)
        # print(pc)
        popt, _ = curve_fit(corey, Sw, pc, bounds=param_bounds_corey)
        pd, pc_lambda = popt
        corey_params[well_id] = (pd, pc_lambda)
    return corey_params


def plot_corey_curve_params(corey_params, grouped_data):
    for well_id, (pd, pc_lambda) in corey_params.items():
        well_data = grouped_data.get_group(well_id)
        Sw = np.array(well_data["Sw"]) / 100
        pc = np.array(well_data["Pc"])
        fig, ax = plt.subplots()
        ax.scatter(Sw, pc, label='Data')
        # Generate points on the Corey curve
        s_values = np.linspace(np.min(Sw), np.max(Sw), 100)
        krw_model = corey(s_values, pd, pc_lambda)
        ax.plot(s_values, krw_model, label='Corey Model')
        ax.set_xlabel('Sw')
        ax.set_ylabel('Pc')
        ax.legend()
        ax.set_title(f'well_id: {well_id}')
        ax.set_xlim([0,1])
        st.pyplot(fig)

@st.cache
def convert_results(corey_params):
    # Convert the dictionary to a DataFrame
    corey_params_df = pd.DataFrame.from_dict(corey_params, orient='index', columns=["pd", "pc_lambda"])
    # Save the DataFrame to a CSV file
    return corey_params_df.to_csv().encode('utf-8')

st.title("Pc Corey fitting")
st.write("Units for Pc are the same as your input data")
c1, c2 = st.columns(2)
with st.container():
    exponent_lb = c1.number_input("Corey exponent lower boundary:", value = 0.5)
    exponent_ub = c2.number_input("Corey exponent upper boundary:", value = 2.0)
c3, c4 = st.columns(2)
with st.container():
    pd_lb = c3.number_input("Entry pressure lower boundary:", value = 0.01)
    pd_ub = c4.number_input("Entry pressure upper boundary:", value = 0.05)
    
    
file = st.file_uploader("Upload Excel file", type="xlsx")
if file is not None:
    grouped_data = data_clean(file)
    corey_params = get_corey_params(grouped_data, pd_lb, pd_ub, exponent_lb, exponent_ub)
    csv = convert_results(corey_params)
    st.download_button(
        label="Download parameters as CSV",
        data=csv,
        file_name='corey_params_water.csv',
        mime='text/csv',
    )
    plot_corey_curve_params(corey_params, grouped_data)