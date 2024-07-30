
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
from scipy.stats import norm
from utils.basic_multi import multi_plotter
from utils.basic_single import single_plotter
import seaborn as sns

try:
    with st.sidebar:
        st.write("### Input Parameters")
        fx = st.checkbox("FX")
        option = st.selectbox("Option Type",
                            ("Long Call", 
                            "Long Put", 
                            "Short Call", 
                            "Short Put",
                            "Multi-Select"))
        if option == "Multi-Select":
            options = st.multiselect(
                "Options",
                ["Long Call", 
                "Long Call",
                "Long Call", 
                "Long Call",
                "Long Put", 
                "Long Put", 
                "Long Put", 
                "Long Put", 
                "Short Call", 
                "Short Call", 
                "Short Call", 
                "Short Call", 
                "Short Put",
                "Short Put",
                "Short Put",
                "Short Put",]
            )

        if option != "Multi-Select":
            cols = st.columns(1)
            with cols[0]:
                volatility = st.number_input("Volatility [%]", min_value=0.0, value=50.0, key='vol')
                strike = st.number_input("Option Strike", min_value=0.0, value=100.0, key='strike')
                spot = st.number_input("Underlying Price", min_value=0.0, value=100.0, key='spot')
                risk_free = st.number_input("Risk Free Rate [%]", value=0.0, key='rf')
                dividends = st.number_input("Dividends [%]", min_value=0.0, value=0.0, key='dividends')
                time = st.number_input("Time [Days]", min_value=1, value=1, key='time')
                if fx:
                    notional = st.number_input("Notional, Base Currency", min_value=1, value=100, key='notional')
                else:
                    position = st.number_input("Num Contracts", min_value=1, value=1, key='pos')
        else:
                cols = st.columns(len(options))
                for i in range(len(cols)):
                    with cols[i]:
                        volatility = st.number_input(f"Volatility [%] for {options[i]}", min_value=0.0, value=50.0, key=f"{options[i]}{i} + vol")
                        strike = st.number_input(f"Option Strike for {options[i]}", min_value=0.0, value=100.0, key=f"{options[i]}{i} + strike")
                        spot = st.number_input(f"Underlying Price for {options[i]}", min_value=0.0, value=100.0, key=f"{options[i]}{i} + spot")
                        risk_free = st.number_input(f"Risk Free Rate for {options[i]} [%]", value=0.0, key=f"{options[i]}{i} + rf")
                        dividends = st.number_input(f"Dividends for {options[i]} [%]", min_value=0.0, value=0.0, key=f"{options[i]}{i} + dividends")
                        time = st.number_input(f"Time for {options[i]} [Days]", min_value=1, value=1, key=f"{options[i]}{i} + time")
                        if fx:
                            notional = st.number_input("Notional, Base Currency", min_value=1, value=100, key=f"{options[i]}{i} + notional")
                        else:
                            position = st.number_input("Num Contracts", min_value=1, value=1, key=f"{options[i]}{i} + pos")

    def d12(volatility, strike, spot, risk_free, dividends, time):
        d_1 = (np.log(spot / strike) + ((risk_free / 100 - dividends / 100) + ((volatility / 100) ** 2) / 2) * (time / 365)) / ((volatility / 100) * np.sqrt(time / 365))
        d_2 = d_1 - (volatility / 100) * np.sqrt(time / 365)
        return d_1, d_2
            


    def get_prices(volatility, strike, spot, risk_free, dividends, time):
        d_1, d_2 = d12(volatility, strike, spot, risk_free, dividends, time)
        call_price = spot * np.exp((-dividends / 100) * (time / 365)) * norm.cdf(d_1) - strike * np.exp((-risk_free / 100) * (time / 365)) * norm.cdf(d_2)
        put_price = strike * np.exp((-risk_free / 100) * (time / 365)) * norm.cdf(-d_2) - spot * np.exp((-dividends / 100) * (time / 365)) * norm.cdf(-d_1)

        if call_price < 0:
            call_price = 0

        if put_price < 0:
            put_price = 0
        
        return call_price, put_price

    def plot_option(options, strike, spot, spot_range):
        if option != "Multi-Select":
            options = options[0]
            if fx:
                notionals=[notional]
                call_price, put_price = get_prices(     st.session_state["vol"],
                                                    st.session_state["strike"],
                                                    st.session_state["spot"],
                                                    st.session_state["rf"],
                                                    st.session_state["dividends"],
                                                    st.session_state["time"],
                                                    st.session_state["notional"])
            else:
                notionals=[1]
                call_price, put_price = get_prices(     st.session_state["vol"],
                                                    st.session_state["strike"],
                                                    st.session_state["spot"],
                                                    st.session_state["rf"],
                                                    st.session_state["dividends"],
                                                    st.session_state["time"],
                                                    notional=None,)
            if "Put" in options:
                if "Long" in options:
                        st.pyplot(single_plotter(spot=spot, strike=strike, op_type='p', tr_type='b', op_pr=call_price, spot_range=spot_range))
                elif "Put" in options:
                        st.pyplot(single_plotter(spot=spot, strike=strike, op_type='p', tr_type='s', op_pr=call_price, spot_range=spot_range))
            elif "Call" in options:
                if "Long" in options:
                        st.pyplot(single_plotter(spot=spot, strike=strike, op_type='c', tr_type='b', op_pr=call_price, spot_range=spot_range))
                elif "Short" in options: 
                        st.pyplot(single_plotter(spot=spot, strike=strike, op_type='c', tr_type='s', op_pr=call_price, spot_range=spot_range))
        else:
            op_list = []
            notionals= []
            for i in range(len(options)):
                notionals.append(st.session_state[f"{options[i]}{i} + notional"] / 10)
                call_price, put_price = get_prices( st.session_state[f"{options[i]}{i} + vol"],
                                                    st.session_state[f"{options[i]}{i} + strike"],
                                                    st.session_state[f"{options[i]}{i} + spot"],
                                                    st.session_state[f"{options[i]}{i} + rf"],
                                                    st.session_state[f"{options[i]}{i} + dividends"],
                                                    st.session_state[f"{options[i]}{i} + time"],
                                                )
                if "Put" in options[i]:
                    if "Long" in options[i]:
                            op_list.append({"op_type": "p", "strike": st.session_state[f"{options[i]}{i} + strike"], 'tr_type': "b", "op_pr": put_price})
                    elif "Short" in options[i]:
                            op_list.append({"op_type": "p", "strike": st.session_state[f"{options[i]}{i} + strike"], 'tr_type': "s", "op_pr": put_price})
                elif "Call" in options[i]:
                    if "Long" in options[i]:
                            op_list.append({"op_type": "c", "strike": st.session_state[f"{options[i]}{i} + strike"], 'tr_type': "b", "op_pr": call_price})
                    elif "Short" in options[i]:
                            op_list.append({"op_type": "c", "strike": st.session_state[f"{options[i]}{i} + strike"], 'tr_type': "s", "op_pr": call_price})
            if fx:
                st.pyplot(multi_plotter(spot=spot, notionals=notionals, op_list=op_list, spot_range=spot_range, fx=fx))

    from streamlit_elements import elements, mui, html

    def greeks(option, volatility, strike, spot, risk_free, dividends, time, weight):
        d_1, d_2 = d12(volatility, strike, spot, risk_free, dividends, time)
        sqrt_t = np.sqrt(time / 365)
        exp_r_d_t = np.exp(-dividends * time / 365)
        exp_r_t = np.exp(-risk_free * time / 365)
        
        gamma = (norm.pdf(d_1)) / (spot * sqrt_t * volatility / 100)
        vega = (spot * sqrt_t * norm.pdf(d_1)) / 100
        
        if "Long" in option:
            if "Call" in option:
                delta = exp_r_d_t * norm.cdf(d_1)
                theta = (-spot * volatility / 100 * norm.pdf(d_1) / (2 * sqrt_t)
                        - risk_free * strike * exp_r_t * norm.cdf(d_2)) / 365
                rho = (strike * time / 365 * exp_r_t * norm.cdf(d_2)) / 10000
            else:
                delta = -exp_r_d_t * norm.cdf(-d_1)
                theta = (-spot * volatility / 100 * norm.pdf(d_1) / (2 * sqrt_t)
                        + risk_free * strike * exp_r_t * norm.cdf(-d_2)) / 365
                rho = -(strike * time / 365 * exp_r_t * norm.cdf(-d_2)) / 10000
        else:  
            gamma = -gamma
            vega = -vega
            if "Call" in option:
                delta = -exp_r_d_t * norm.cdf(d_1)
                theta = -(-spot * volatility / 100 * norm.pdf(d_1) / (2 * sqrt_t)
                        + risk_free * strike * exp_r_t * norm.cdf(d_2)) / 365
                rho = -(strike * time / 365 * exp_r_t * norm.cdf(d_2)) / 10000
            else:  
                delta = exp_r_d_t * norm.cdf(-d_1)
                theta = -((spot * volatility / 100 * norm.pdf(d_1) / (2 * sqrt_t) - risk_free * strike * exp_r_t * norm.cdf(-d_2)) / 365)
                rho = (strike * time / 365 * exp_r_t * norm.cdf(-d_2)) / 10000

        greeks_values = [delta, gamma, vega, theta, rho]
        return [weight * x for x in greeks_values]

    def plot_greeks(options, multi=True):
        global greeks
        df_list = []
        x = len(options)
        cols = st.columns(x)
        for i in range(x):
            with cols[i]:
                if multi:
                    ts = [i for i in range(1, st.session_state[f"{options[i]}{i} + time"])]
                else:
                    ts = [i for i in range(1, st.session_state["time"])]
                deltas = []
                gammas = []
                vegas = []
                thetas = []
                rhos = []
                for time in ts:
                    if multi:
                        temp = greeks(options[i], st.session_state[f"{options[i]}{i} + vol"],
                                                        st.session_state[f"{options[i]}{i} + strike"],
                                                        st.session_state[f"{options[i]}{i} + spot"],
                                                        st.session_state[f"{options[i]}{i} + rf"],
                                                        st.session_state[f"{options[i]}{i} + dividends"],
                                                        time,
                                                        st.session_state[f"{options[i]}{i} + notional"] / 10)
                    else:
                        temp = greeks(options[i], st.session_state["vol"],
                                                        st.session_state["strike"],
                                                        st.session_state["spot"],
                                                        st.session_state["rf"],
                                                        st.session_state["dividends"],
                                                        time,
                                                        st.session_state["pos"])
                    deltas.append(temp[0])
                    gammas.append(temp[1])
                    vegas.append(temp[2])
                    thetas.append(temp[3])
                    rhos.append(temp[4])

                greek_df = pd.DataFrame.from_dict({"Delta": deltas, "Gamma": gammas, "Vega": vegas,
                                                "Theta": thetas, "Rho": rhos})
                df_list.append(greek_df)
                fig, _ = plt.subplots(5, 1, figsize=(10, 12))
                for j, ax in enumerate(fig.axes):
                    ax.plot(ts, greek_df.iloc[:, j], label=greek_df.columns[j], linestyle="--")
                    ax.set_title(greek_df.columns[j])
                    ax.invert_xaxis()
                if multi:
                    fig.suptitle(f"{options[i]} ST: {st.session_state[f'{options[i]}{i} + strike']}", fontsize=16)
                else:
                    fig.suptitle(f"{options[i]} ST: {st.session_state[f'strike']}", fontsize=16)
                fig.subplots_adjust(hspace=0.3)
                st.pyplot(fig)
                
    df = pd.DataFrame([], index=["Delta", "Gamma", "Vega", "Theta (/Trading Day)", "Rho (/bp)"])

    if option != "Multi-Select":
        with elements(key=option):
            call_price, put_price = get_prices(volatility, strike, spot, risk_free, dividends, time)
            if fx:
                put_price *= notional 
                call_price *= notional
            if "Put" in option:
                mui.Box(f"{option} Premium: ${put_price:.3f}",
                    sx={
                        "fontSize": 20,
                        "bgcolor": "background.paper",
                        "boxShadow": 1,
                        "borderRadius": 0,
                        "p": 5,
                        "minWidth": 50,
                        "width": 260,
                        "fontWeight": 'bold',
                        "textAlign": 'center',
                        "fontFamily": 'Monospace',
                        "color": 'text.primary',
                        "margin-left": 'auto',
                        "margin-right": 'auto',
                    })
            elif "Call" in option:
                mui.Box(f"{option} Premium: ${call_price:.3f}",
                        sx={
                            "fontSize": 20,
                            "bgcolor": "background.paper",
                            "boxShadow": 1,
                            "borderRadius": 0,
                            "p": 5,
                            "minWidth": 50,
                            "width": 260,
                            "fontWeight": 'bold',
                            "textAlign": 'center',
                            "fontFamily": 'Monospace',
                            "color": 'text.primary',
                            "margin-left": 'auto',
                            "margin-right": 'auto'
                    })      
        if fx:
            temp = pd.DataFrame(greeks(option, volatility, strike, spot, risk_free, dividends, time, notional), columns=[option], index=["Delta", "Gamma", "Vega", "Theta (/Trading Day)", "Rho (/bp)"])
        else:             
            temp = pd.DataFrame(greeks(option, volatility, strike, spot, risk_free, dividends, time, position), columns=[option], index=["Delta", "Gamma", "Vega", "Theta (/Trading Day)", "Rho (/bp)"])
        df = pd.concat([df, temp], axis=1).round(2)
        st.table(df)

        spot_range = st.number_input("Spot Range", value=10)
        plot_option([option], strike, spot, spot_range)
        pg = st.checkbox("Plot Greeks")
        if pg:
            plot_greeks([option], multi=False)
        heatmap = st.checkbox("Heatmap")
        if heatmap:
            vol_range = st.slider("Volatility Range [%]", min_value=1, max_value=10)

            lower_vol = st.session_state['vol'] - vol_range
            upper_vol = st.session_state['vol'] + vol_range
            if lower_vol < 0:
                lower_vol = 0

            vol_range = list(np.arange(lower_vol, upper_vol, 2 * vol_range / 10))
            spot_range = list(np.arange(spot - spot_range, spot + spot_range, 2 * spot_range / 10))
            df_puts = pd.DataFrame(index=vol_range, columns=spot_range)
            df_calls = pd.DataFrame(index=vol_range, columns=spot_range)
            for vol in vol_range:
                for spot in spot_range:
                    call_price, put_price = get_prices(vol,
                                                    st.session_state["strike"],
                                                    spot,
                                                    st.session_state["rf"],
                                                    st.session_state["dividends"],
                                                    st.session_state["time"],
                                                    st.session_state[f"{options[i]}{i} + notional"])
                    
                    df_puts.loc[vol, spot] = put_price 
                    df_calls.loc[vol, spot] = call_price
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
            df_calls.dropna(inplace=True)
            df_puts.dropna(inplace=True)
            df_puts = df_puts.round(2)
            df_calls = df_calls.round(2)
            df_puts.index = df_puts.index.round(2)
            df_calls.index = df_calls.index.round(2)
            put_heatmap = sns.heatmap(df_puts.astype('float'), cmap='crest', annot=True, ax=ax1)
            call_heatmap = sns.heatmap(df_calls.astype('float'), cmap='crest', annot=True, ax=ax2)
            
            call_heatmap.set_ylabel('Vol')
            put_heatmap.set_ylabel('Vol')
            call_heatmap.set_xlabel('Spot')
            put_heatmap.set_xlabel('Spot')
            plt.subplots_adjust(hspace=0.2)
            st.pyplot(fig)
    else:
            with elements(key=options):
                df = pd.DataFrame(index=["Delta", "Gamma", "Vega", "Theta (/Trading Day)", "Rho (/bp)"])
                for i in range(len(options)):
                    call_price, put_price = get_prices(st.session_state[f"{options[i]}{i} + vol"],
                                                    st.session_state[f"{options[i]}{i} + strike"],
                                                    st.session_state[f"{options[i]}{i} + spot"],
                                                    st.session_state[f"{options[i]}{i} + rf"],
                                                    st.session_state[f"{options[i]}{i} + dividends"],
                                                    st.session_state[f"{options[i]}{i} + time"],
                                                    )
                    if fx:
                        put_price *= notional / 10
                        call_price *= notional / 10
                    if "Call" in options[i]:
                        mui.Box(
                                f"{options[i]} Premium: ${call_price:.3f}",
                                sx={"display": 'inline-grid',
                                    "fontSize": 18,
                                    "bgcolor": "background.paper",
                                    "boxShadow": 1,
                                    "borderRadius": 0,
                                    "p": 5,
                                    "minWidth": 50,
                                    "width": 260,
                                    "fontWeight": 'bold',
                                    "textAlign": 'center',
                                    "fontFamily": 'Monospace',
                                    "color": 'text.primary'
                                }    
                            )
                    else:
                        mui.Box(
                                f"{options[i]} Premium: ${put_price:.3f}",
                                sx={"display": 'inline-grid',
                                    "fontSize": 18,
                                    "bgcolor": "background.paper",
                                    "boxShadow": 1,
                                    "borderRadius": 0,
                                    "p": 5,
                                    "minWidth": 50,
                                    "width": 260,
                                    "fontWeight": 'bold',
                                    "textAlign": 'center',
                                    "fontFamily": 'Monospace',
                                    "color": 'text.primary'
                                }      
                            )
                    temp = pd.DataFrame(greeks(options[i], st.session_state[f"{options[i]}{i} + vol"],
                                                    st.session_state[f"{options[i]}{i} + strike"],
                                                    st.session_state[f"{options[i]}{i} + spot"],
                                                    st.session_state[f"{options[i]}{i} + rf"],
                                                    st.session_state[f"{options[i]}{i} + dividends"],
                                                    st.session_state[f"{options[i]}{i} + time"],
                                                    st.session_state[f"{options[i]}{i} + notional"] / 10), columns=[f"{options[i]} ST: {st.session_state[f'{options[i]}{i} + strike']}"], 
                                                    index=["Delta", "Gamma", "Vega", "Theta (/Trading Day)", "Rho (/bp)"])
                    df = pd.concat([df, temp], axis=1).round(2)
            df["Cumulative Exposure"] = df.sum(axis=1)
            st.table(df)
            spot_range = st.number_input("Spot Range", value=10)
            plot_option(options, strike, spot, spot_range)
            pg = st.checkbox("Plot Greeks")
            if pg:
                plot_greeks(options)
            heatmap = st.checkbox("Heatmap")


except:
    st.write("### Nothing to see here")