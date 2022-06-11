import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from util import get_data
import scipy.optimize as spo


def optimize_portfolio(
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 1, 1),
        syms=["GOOG", "AAPL", "GLD", "XOM"],
        gen_plot=False,
):
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe  		  	   		   	 		  		  		    	 		 		   		 		  
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of  		  	   		   	 		  		  		    	 		 		   		 		  
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take  		  	   		   	 		  		  		    	 		 		   		 		  
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and  		  	   		   	 		  		  		    	 		 		   		 		  
    statistics.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		   	 		  		  		    	 		 		   		 		  
    :type sd: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		   	 		  		  		    	 		 		   		 		  
    :type ed: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
    :param syms: A list of symbols that make up the portfolio (note that your code should support any  		  	   		   	 		  		  		    	 		 		   		 		  
        symbol in the data directory)  		  	   		   	 		  		  		    	 		 		   		 		  
    :type syms: list  		  	   		   	 		  		  		    	 		 		   		 		  
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your  		  	   		   	 		  		  		    	 		 		   		 		  
        code with gen_plot = False.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type gen_plot: bool  		  	   		   	 		  		  		    	 		 		   		 		  
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,  		  	   		   	 		  		  		    	 		 		   		 		  
        standard deviation of daily returns, and Sharpe ratio  		  	   		   	 		  		  		    	 		 		   		 		  
    :rtype: tuple  		  	   		   	 		  		  		    	 		 		   		 		  
    """

    # Read in adjusted closing prices for given symbols, date range  		  	   		   	 		  		  		    	 		 		   		 		  
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY  		  	   		   	 		  		  		    	 		 		   		 		  
    prices = prices_all[syms]  # only portfolio symbols  		  	   		   	 		  		  		    	 		 		   		 		  
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later

    fill_missing_values(prices)

    # find the allocations for the optimal portfolio
    # see lecture 01-08 Optimizers: Building a parameterized model
    n = len(syms)
    init_allocs = np.ones((1, n)) / n
    ranges = [(0.0, 1.0)] * n
    constraints = ({'type': 'eq', 'fun': lambda inputs: np.sum(inputs) - 1})
    result = spo.minimize(minimize_func,
                          init_allocs,
                          args=prices,
                          method='SLSQP',
                          bounds=ranges,
                          constraints=constraints)
    allocs = result.x

    # compute stats
    port_val = get_daily_portfolio_val(prices, allocs)
    cr = get_cumulative_return(port_val)
    daily_return = get_daily_return(port_val)
    adr = daily_return.mean()
    sddr = daily_return.std()
    sr = get_sharpe_ratio(adr, sddr)

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:

        prices_SPY = prices_SPY / prices_SPY[0]  # normalized SPY

        df_temp = pd.concat(
            [port_val, prices_SPY], keys=["Portfolio", "SPY"], axis=1
        )
        df_temp.columns = ['Portfolio', 'SPY']
        plot_data(df_temp,
                  title='Daily Portfolio Value and SPY',
                  xlabel='Date',
                  ylabel='Normalized Price')

        pass

    return allocs, cr, adr, sddr, sr


def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    """
        Plot stock prices with a custom title and meaningful axis labels.
        
        Copy from util and add save fig.
    """
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig('Figure1.png')
    # plt.show()


def minimize_func(allocs, prices):
    """
        This function calculate the negative Sharpe ratio.

        See lecture 01-07 Sharpe ratio and other portfolio statistics
    """
    port_val = get_daily_portfolio_val(prices, allocs)
    daily_return = get_daily_return(port_val)
    adr = daily_return.mean()
    sddr = daily_return.std()
    sr = get_sharpe_ratio(adr, sddr)

    return -sr


def fill_missing_values(df_data):
    """
        Fill missing values in data frame, in place.

        Lecture 01-05 Incomplete Data
    """
    # pass
    df_data.fillna(method="ffill", inplace=True)
    df_data.fillna(method="bfill", inplace=False)


def get_daily_portfolio_val(prices, allocs):
    """
        This function calculate the daily portfolio value.

        price -> normed -> alloced -> pos_vals -> port_val

        See lecture 01-07 Sharpe ratio and other portfolio statistics
    """
    # normalize
    normed = prices / prices.ix[0]
    # position values
    pos_vals = allocs * normed
    # sum each row to get value of the portfolio on each corresponding day
    port_val = pos_vals.sum(axis=1)

    return port_val


def get_daily_return(port_val):
    """
        Compute and return the daily return values, i.e. how much the price go up or down on a
        particular day. Daily return of day t is the price on day t subtract the price of the
        previous day subtract one.

        daily_ret[t] = (price[t]/price[t-1]) - 1

        Lecture 01-04 Statistical analysis of time series
    """
    # daily_returns = port_val.copy()
    # daily_returns[1:] = (port_val[1:] / port_val[:-1].values) - 1 # compute daily returns for row 1 onwards
    daily_returns = (port_val / port_val.shift(1)) - 1  # much easier with Pandas!
    # daily_returns.iloc[0, :] = 0  # Pandas leaves the 0th row full of Nans (errors when leave in)

    # skip the first row
    daily_returns = daily_returns[1:]

    return daily_returns


def get_cumulative_return(port_val):
    """
        Compute and return the cumulative return values. It is the price at the end, divided by
        the price at the beginning, minus one.

        cum_ret = (price[t] / price[0]) - 1

        Lecture 01-04 Statistical analysis of time series
    """
    return port_val[-1] / port_val[0] - 1


def get_sharpe_ratio(mean_daily_rets, std_daily_rets):
    """
        Calculate and return the Sharpe Ratio, a metric that adjusts return for risk.

        Rp = portfolio return
        Rf = risk free rate of return (usually 0%)
        Sigma_p = std of portfolio return

        SR = mean(Rp - Rf) / Std(Rp - Rf)
        SR = mean(daily_rets - daily_rf) / std(daily_rets - daily_rf)
        SR = mean(daily_rets) / std(daily_rets)

        There are 252 trading days per year
            daily k = sqrt(252)
            weekly k = sqrt(52)
            onthly k = sqrt(12)
        SR_annualized = k * SR

        Lecture 01-07 Sharpe ratio and other portfolio statistics
    """

    k = 252
    return (k ** 0.5) * mean_daily_rets / std_daily_rets


def test_code():
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    This function WILL NOT be called by the auto grader.  		  	   		   	 		  		  		    	 		 		   		 		  
    """

    # start_date = dt.datetime(2009, 1, 1)
    # end_date = dt.datetime(2010, 1, 1)
    # symbols = ["GOOG", "AAPL", "GLD", "XOM", "IBM"]
    #
    # # Assess the portfolio
    # allocations, cr, adr, sddr, sr = optimize_portfolio(
    #     sd=start_date, ed=end_date, syms=symbols, gen_plot=True
    # )

    start_date = dt.datetime(2008, 6, 1)
    end_date = dt.datetime(2009, 6, 1)
    symbols = ['IBM', 'X', 'GLD', 'JPM']
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd=start_date,
                                                        ed=end_date,
                                                        syms=symbols,
                                                        gen_plot=True)

    # Print statistics  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Symbols: {symbols}")
    print(f"Allocations:{allocations}")
    print(f"Sharpe Ratio: {sr}")
    print(f"Volatility (stdev of daily returns): {sddr}")
    print(f"Average Daily Return: {adr}")
    print(f"Cumulative Return: {cr}")


if __name__ == "__main__":
    test_code()
