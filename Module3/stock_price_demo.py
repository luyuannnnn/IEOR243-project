import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from streamlit_modal import Modal
from ipywidgets import interact, widgets, VBox, HBox
from IPython.display import display


# Company Info
wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
cik_df = pd.read_html(wiki_url, header=0, index_col=0)[0]
cik_df['GICS Sector'] = cik_df['GICS Sector'].astype("category")
target_ciks = [320193, 789019, 1018724, 66740, 766704, 1067983, 200406, 1800, 104169, 19617,
               80424, 109380, 731766, 354950, 1467373, 1045810, 796343, 1744489, 1174922, 1065280,
               50863, 21344, 77476, 93410, 34088, 858877, 1341439, 310158, 78003, 72971]
cik_df['CIK'] = cik_df['CIK'].astype(str).str.zfill(10).astype(int)
filtered_cik_df = cik_df[cik_df['CIK'].isin(target_ciks)]
filtered_cik_df = filtered_cik_df.reset_index()
filtered_cik_df.drop(columns=["Date added"],inplace=True)
new_row = pd.DataFrame({'Symbol': ["ZION"], 'Security': ["Zions Bancorporation"], "GICS Sector": ["Financials"], "GICS Sub-Industry": ["Regional Banks"], "Headquarters Location": ["Salt Lake City, Utah"], "CIK": [109380], "Founded": ["1873"]})
filtered_cik_df = pd.concat([filtered_cik_df, new_row], ignore_index=True)

ticker_list = "AAPL ABT ACN ADBE AMZN BRK-B CSCO CVX DIS HD INTC JNJ JPM KO MMM MRK MSFT NFLX NVDA ORCL PEP PFE PG UNH WELL WFC WMT WYNN XOM ZION"
company_names = ['Apple Inc.', 'Abbott Laboratories', 'ACCENTURE PLC', 'Adobe Inc.', 'Amazon.com Inc.', 
                 'Berkshire Hathaway Inc.', 'Cisco Systems Inc.', 'Chevron Corporation', 'Walt Disney Company', 'Home Depot Inc.', 
                 'Intel Corporation', 'Johnson & Johnson', 'JPMorgan Chase & Co.', 'Coca-Cola Company', '3M CO', 
                 'Merck & Co. Inc.', 'Microsoft Corporation', 'Netflix Inc.', 'NVIDIA Corporation', 'Oracle Corporation', 
                 'PepsiCo Inc.', 'Pfizer Inc.', 'Procter & Gamble Co.', 'UnitedHealth Group Incorporated', 'Welltower Inc.', 
                 'Wells Fargo & Company', 'Walmart Inc.', 'Wynn Resorts Limited', 'Exxon Mobil Corporation', 'Zions Bancorporation National Association']
Ticker_CIKs_mapping = {'AAPL':'320193','MSFT':'789019','AMZN':'1018724','MMM':'66740','WELL':'766704',
    'BRK-B':'1067983','JNJ':'200406','ABT':'1800','WMT':'104169','JPM':'19617',
    'PG':'80424','ZION':'109380','UNH':'731766','HD':'354950','ACN':'1467373',
    'NVDA':'1045810','ADBE':'796343','DIS':'1744489','WYNN':'1174922','NFLX':'1065280',
    'INTC':'50863','KO':'21344','PEP':'77476','CVX':'93410','XOM':'34088',
    'CSCO':'858877','ORCL':'1341439','MRK':'310158','PFE':'78003','WFC':'72971'}


# -----------------Tab--------------------------------------------
listTabs = ["📈 Stock Performance", "💡 Stock Recommendation", "🔍 Monitor Watchlist"]
whitespace = 27
tab1, tab2, tab3= st.tabs([s.center(whitespace, "\u2001" ) for s in listTabs] )


# -----------------Sidebar: Add watchlist --------------------------------------------
watchlist = []
with st.sidebar:
    st.header("Watchlist")
    st.write("If you want to add the company to watchlist, please select it!")
    for company in company_names:
        if st.sidebar.checkbox(company):
            watchlist.append(company)

    st.markdown("----")
    
    st.header('Feedback')
    feedback = st.text_area("Please enter your feedback or suggestions:", "")

    # Add a submit button
    if st.button('Submit'):
        # Handle the submitted feedback
        if feedback:
            st.success('Thank you for your feedback! We will carefully consider your suggestions.')
        else:
            st.warning('Please enter your feedback before submitting.')

       

# -----------------Part1 Stock Performance by Day ------------------------------------
with tab1:
    # Disclaimer
    if 'clicked' not in st.session_state:
        st.session_state.clicked = False
    def click_clicked():
        st.session_state.clicked = True

    risk_disclaimer = Modal(title="Welcome to Stock Analysis app!", key="risk_disclaimer", max_width=800)
    if st.session_state.clicked == False:
        with risk_disclaimer.container():
            st.markdown("Please be aware that predictions are for informational purposes only and should not be considered as investment advice. The stock market can be unpredictable, and predictions may not always be accurate. Before investing, it's important to do your own research and understand the risks involved. Remember that investing in stocks carries the risk of potential loss.")
            if st.button("I understand", key="understand", on_click=click_clicked):
                risk_disclaimer.close()
                st.experimental_rerun()


    # stock performance
    st.header("Stock Performance by Day")
    stock_price_day = yf.download(ticker_list, start="2003-01-01",interval="1d")
    close_stock_price_day = stock_price_day["Close"]
    min_date = close_stock_price_day.index.min()
    max_date = close_stock_price_day.index.max()
    close_stock_price_day.columns = company_names

    # Create date slider
    date_range_indices = np.arange(len(close_stock_price_day))

    date_slider = st.date_input("Date Range", min_value=min_date, max_value=max_date, value=(min_date, max_date))

    # Create dropdown for company selection
    company_dropdown = st.selectbox("Company Name", close_stock_price_day.columns)

    # Create dropdown for moving average length
    ma_length_dropdown = st.selectbox("Moving Average Length", [100, 200, 300, 500, 800])
    st.markdown("*Note: Moving average length is to calculate the average stock price of the input days. The red line in the graph reveals underlying trends over time.*")

    # Time series plot with moving average
    def plot_stock_price(company, ma_length, date_range):
        start_date, end_date = date_range
        subset = close_stock_price_day.loc[start_date:end_date]

        # Calculate the moving average
        subset['MA'] = subset[company].rolling(window=ma_length).mean()

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(subset.index, subset[company], label=f'{company} Stock Price')
        ax.plot(subset.index, subset['MA'], label=f'{company} {ma_length}-Day Moving Average', color='red')

        # Add labels and legend
        ax.set_title(f'Stock Price Trend for {company}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Close Stock Price')
        ax.tick_params(axis='x', rotation=45)
        ax.legend()

        # Show plot
        st.pyplot(fig)
        

    # Interactive widgets
    plot_stock_price(company_dropdown, ma_length_dropdown, date_slider)


# -----------------Part2 Stock Recomendation ------------------------------------
with tab2:
    st.header("Stock Recommendation")

    reco_method = st.radio(
        "Select a method for recommendation",
        ( "Automatical Recommendation", "Custom Condition")
    )

    if reco_method == "Automatical Recommendation":

        stock_price_quarter = yf.download(ticker_list, start="2019-01-01", end = '2023-12-31',interval="3mo")
        close_stock_price_quarter = stock_price_quarter["Close"]
        close_stock_price_quarter = close_stock_price_quarter.reset_index(drop=False)

        result = pd.read_csv("svr_result.csv")
        result_last = result[(result['Year'] == 2023) & (result['Quarter'] == 'Q4')]
        result_last['2024-01-01'] = np.exp(result_last["LogClosePriceNextQuarter"])
        result_last = result_last.set_index('Ticker')
        result_last = result_last.transpose().rename_axis('Date').iloc[-1:]
        result_last = result_last.reset_index(drop=False)
        close_stock_price_pred = pd.concat([close_stock_price_quarter, result_last])
        close_stock_price_pred = close_stock_price_pred.set_index('Date') 

        # calculate growth rate of the last two months
        last_two_quarter = close_stock_price_pred.tail(2)
        last_two_quarter = last_two_quarter.transpose()
        last_two_quarter = last_two_quarter.reset_index()
        last_two_quarter["Growth Rate"] = (last_two_quarter.iloc[:,-1] - last_two_quarter.iloc[:,-2])/last_two_quarter.iloc[:,-2]
        last_two_quarter['CIKs'] = last_two_quarter['Ticker'].map(Ticker_CIKs_mapping)
        last_two_quarter['CIKs'] = last_two_quarter['CIKs'].astype('int')
        last_two_quarter = pd.merge(last_two_quarter, filtered_cik_df , left_on=['CIKs'], right_on = ['CIK'], how='left')

        
        # present the company with satisfied growth rate
        st.markdown("#### Recommended Company List based on Predicted Growth Rate")
        growth_rate_threshold = st.number_input('Growth rate (%) of stock price should be equal to or larger than', min_value=0.0, max_value=100.0, value=5.0)/100
        gics_sector_list = last_two_quarter['GICS Sector'].unique()
        gics_sector = st.selectbox('Select GICS Sector', gics_sector_list)

        filtered_data = last_two_quarter[(last_two_quarter['Growth Rate'] >= growth_rate_threshold / 100) & (last_two_quarter['GICS Sector'] == gics_sector)]
        filtered_data = filtered_data.sort_values(by='Growth Rate', ascending=False)
        filtered_data = filtered_data.set_index('Ticker')
        filtered_data['Growth Rate'] = filtered_data['Growth Rate'].map(lambda x: "{:.2%}".format(x))
        features = ['Security', 'Growth Rate','GICS Sector','GICS Sub-Industry', 'Headquarters Location','CIK', 'Founded']
        output_data = filtered_data[features]
        st.write(output_data)


        st.markdown("----")

        st.markdown("#### Stock Performance Prediction by Quarter")
        st.markdown("*Note: The red point in the following graph is the prediction based on model.*")

        close_stock_price_pred.columns = company_names

        # Create date slider
        date_range_indices = np.arange(len(close_stock_price_pred))
        # Create dropdown for company selection
        company_dropdown = st.selectbox("Select a company", close_stock_price_pred.columns,key="company_dropdown2")

        def plot_stock_price_with_progress(company):
            # Get the subset of data for the specified company
            subset = close_stock_price_pred
            subset.index = pd.to_datetime(subset.index)

            # Create a progress bar
            progress_bar = st.progress(0)

            # Plotting
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(subset.index, subset[company], marker='o', label=f'{company} Stock Price by Quarter')
            # Add the last point and mark it as red
            last_date = subset.index[-1]
            last_price = subset[company].iloc[-1]
            ax.plot(last_date, last_price, marker='o', markersize=8, color='red')
            # Add labels and legend
            ax.set_title(f'Stock Price Prediction for {company}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Close Stock Price')
            ax.tick_params(axis='x', rotation=45)
            ax.legend()

            st.pyplot(fig)
            progress_bar.progress(100)

        plot_stock_price_with_progress(company_dropdown)


    if reco_method == "Custom Condition":

        # input data of indicator 
        indicator = pd.read_csv("stock_price_prediction_data.csv")
        indicator['Date'] = indicator.apply(lambda row: f"{row['Year']}-{row['Quarter']}", axis=1)
        indicator['Date'] = pd.to_datetime(indicator['Date'])
        indicator.drop(['Year', 'Quarter','YearNext', 'QuarterNext'], axis=1, inplace=True)
        columns_to_divide = ['FreeCashFlow', 'NetIncome', 'OperatingCashFlow','ClosePriceNextQuarter'] ## million
        indicator[columns_to_divide] = indicator[columns_to_divide] / 1000000 
        columns_to_divide = ['TotalRevenue'] ## billion
        indicator[columns_to_divide] = indicator[columns_to_divide] / 1000000000 
        indicator_last = indicator[indicator['Date'] == "2023-10-01"]


        # Function to plot time series
        def plot_time_series(indicator, CIK, selected_indicator):
            plt.figure(figsize=(10, 6))
            plt.plot(indicator[indicator['CIKs'] == int(CIK)]['Date'], 
                     indicator[indicator['CIKs'] == int(CIK)][selected_indicator])
            plt.xlabel("Date")
            plt.ylabel(selected_indicator)
            st.pyplot(plt)

        st.markdown('#### Custom Condition')
        indicator_list = ["DilutedEPS", "FreeCashFlow", "NetIncome", "OperatingCashFlow", "TotalRevenue", "Polarity", "Subjectivity"]
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            selected_indicator = st.selectbox('Select Indicator', indicator_list)
        with col2:
            condition = st.selectbox('Select Condition', ['>', '=', '<'])
        with col3:
            custom_value = st.number_input('Enter Value')
        with st.expander("Click to see the explanation of indicators"):
            st.write("""
                - **DilutedEPS:** a company's profitability by dividing its net income by the total number of diluted shares outstanding.
                - **FreeCashFlow:** the cash generated by a company's operations after accounting for capital expenditures.
                - **NetIncome:** also known as the bottom line or profit, is the total profit earned by a company after deducting all expenses, taxes, and interest from its total revenue.
                - **OperatingCashFlow:** the cash generated or used by a company's core operating activities, such as sales of goods and services, minus operating expenses.
                - **TotalRevenue:** the overall amount of money generated by a company from its core business activities, including sales of goods and services, interest, and other sources of income.
                - **Polarity:** the sentiment orientation or emotional direction of the text. A value of -1 represents extremely negative sentiment, while a value of 1 represents extremely positive sentiment.
                - **Subjectivity:** the amount of personal opinion versus factual information contained in the text of financial statements(10-Q filings). Higher subjectivity suggests more opinion-based content.
            """)
            st.write("**Reference: The model identifies DilutedEPS as having a high significance with stock price.**")


        # Filter data based on the condition selected by the user
        filtered_data = None
        if condition == '>':
            filtered_data = indicator_last[indicator_last[selected_indicator] > custom_value]
        elif condition == '=':
            filtered_data = indicator_last[indicator_last[selected_indicator] == custom_value]
        elif condition == '<':
            filtered_data = indicator_last[indicator_last[selected_indicator] < custom_value]

        # Display time series plot on company selection
        if not filtered_data.empty:
            selected_company = filtered_data['CIKs'].unique()
            filtered_company_info = filtered_cik_df[filtered_cik_df['CIK'].isin(selected_company)]
            selected_company_name = filtered_cik_df.loc[filtered_cik_df['CIK'].isin(selected_company), 'Security'].iloc[0]  
            # Display company information
            st.write(filtered_company_info)

            # Display time series plot on company click
            company_selection = st.radio("Select a company", filtered_company_info['Security'].unique())
            selected_company_cik = filtered_company_info.loc[filtered_company_info['Security'] == company_selection, 'CIK'].iloc[0]
            selected_company_cik = int(selected_company_cik)
            st.write(f"Time Series Plot for {company_selection}'s {selected_indicator}")
            plot_time_series(indicator = indicator, CIK = selected_company_cik, selected_indicator = selected_indicator)
        else:
            st.write("No company satisfies this condition.")
            
    

# -----------------Part3 Stock Monitoring ------------------------------------
with tab3:
    st.header("Your Watchlist:")
    if len(watchlist) == 0:
        st.write("Your watchlist is currently empty.")
    else:
        for company in watchlist:
            st.markdown(f"#### {company}")
            price_day3 = yf.download(ticker_list, start="2003-01-01",interval="1d")
            price_day3 = price_day3['Close']
            price_day3.columns = company_names
            price_quarter3 = yf.download(ticker_list, start="2003-01-01",interval="3mo")
            price_quarter3 = price_quarter3['Close']
            price_quarter3.columns = company_names

            # Get daily stock price data for the company
            daily_price_data = price_day3.loc[:, company]
            daily_price = daily_price_data.iloc[-1]
            # Get quarterly stock price data for the company
            quarterly_price_data = price_quarter3.loc[:, company]
            quarterly_price = quarterly_price_data.iloc[-2]

            # Calculate day-over-day change
            daily_change = ((daily_price - daily_price_data.iloc[-2]) / daily_price_data.iloc[-2]) * 100

            # Calculate quarter-over-quarter change
            quarterly_change = ((quarterly_price - quarterly_price_data.iloc[-3]) / quarterly_price_data.iloc[-3]) * 100
            
            # Display metrics
            col1, col2 = st.columns(2)
            col1.metric("Daily Price", f"${daily_price:.2f}",f"{daily_change:.2f}%")
            col2.metric("Quarterly Price", f"${quarterly_price:.2f}",f"{quarterly_change:.2f}%")
            st.markdown("----")
