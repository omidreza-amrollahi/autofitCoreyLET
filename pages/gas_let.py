import streamlit as st
import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

def let(s, krg_swc, lg, eg, tg):
    se = (s - np.min(s)) / (1 - np.min(s) - (1 - np.max(s)))
    krg = krg_swc * ( ( np.abs(1 - se) ** lg ) / ( np.abs(1 - se) ** lg + eg * (se ** tg) ) )
    return krg

def data_clean(file):
    df = pd.read_excel(file, header=0, skiprows=[1])
    df = df[["ID PK", "Sw", "Krg"]]
    # remove the rows where krg is above 1
    df = df[df["Krg"] <= 1.0]
    df = df.dropna()
    grouped_data = df.groupby("ID PK")[["Sw", "Krg"]]
    return grouped_data

def get_let_params(grouped_data, L_lb=0.1, L_ub=20, E_lb=0, E_ub=20, T_lb=0.1, T_ub=5):
    let_params = {}
    param_bounds_let = [(L_lb, E_lb, T_lb), (L_ub, E_ub, T_ub)]
    for i, (well_id, well_data) in enumerate(grouped_data):
        # if i > 5:
        #     break
        Sw = np.array(well_data["Sw"]) / 100
        krg = np.array(well_data["Krg"])
        # print(well_id)
        # print(Sw)
        # print(krg)
        krg_swc = np.max(krg)
        if len(Sw)>2:
            try:
                popt, _ = curve_fit(lambda s, lg, eg, tg: let(s, krg_swc, lg, eg, tg), Sw, krg, bounds=param_bounds_let, p0=[1 ,1 ,1])
                lg, eg, tg = popt
                let_params[well_id] = (krg_swc, lg, eg, tg)
            except:
                print("Fitting error!")
                plt.scatter(Sw, krg, label='Data')
                plt.show()
                
        else:
            print(f"The number of func parameters=3 must not exceed the number of data points={len(Sw)}")
    return let_params

def plot_let_curve_params(let_params, grouped_data):
    for well_id, (krg_swc, lg, eg, tg) in let_params.items():
        well_data = grouped_data.get_group(well_id)
        Sw = np.array(well_data["Sw"]) / 100
        krg = np.array(well_data["Krg"])
        fig, ax = plt.subplots()
        ax.scatter(Sw, krg, label='Data')
        # Generate points on the Corey curve
        s_values = np.linspace(np.min(Sw), np.max(Sw), 100)
        krg_model = let(s_values, krg_swc, lg, eg, tg)
        ax.plot(s_values, krg_model, label='LET Model')
        ax.set_xlabel('Sw')
        ax.set_ylabel('krg')
        ax.legend()
        ax.set_title(f'well_id: {well_id}')
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        st.pyplot(fig)

@st.cache
def convert_results(let_params):
    # Convert the dictionary to a DataFrame
    let_params_df = pd.DataFrame.from_dict(let_params, orient='index', columns=["krg_swc", "lg", "eg", "tg"])
    # Save the DataFrame to a CSV file
    return let_params_df.to_csv().encode('utf-8')

st.title("Gas LET fitting")
c1, c2 = st.columns(2)
with st.container():
    L_lb = c1.number_input("L-value lower boundary:", 0.1, 50.0, 0.1)
    L_ub = c2.number_input("L-value upper boundary:", 0.1, 50.0, 20.0)
c3, c4 = st.columns(2)
with st.container():
    E_lb = c3.number_input("E-value lower boundary:", 0.0, 50.0, 0.0)
    E_ub = c4.number_input("E-value upper boundary:", 0.0, 50.0, 20.0)
c5, c6 = st.columns(2)
with st.container():
    T_lb = c5.number_input("T-value lower boundary:", 0.1, 50.0, 0.1)
    T_ub = c6.number_input("T-value upper boundary:", 0.1, 50.0, 5.0)


file = st.file_uploader("Upload Excel file", type="xlsx")
if file is not None:
    grouped_data = data_clean(file)
    let_params = get_let_params(grouped_data, L_lb, L_ub, E_lb, E_ub, T_lb, T_ub)
    csv = convert_results(let_params)
    st.download_button(
        label="Download parameters as CSV",
        data=csv,
        file_name='let_params_gas.csv',
        mime='text/csv',
    )
    plot_let_curve_params(let_params, grouped_data)