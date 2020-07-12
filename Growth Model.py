import quantopian.algorithm as algo
import quantopian.optimize as opt
from quantopian.pipeline import Pipeline
from quantopian.pipeline.factors import SimpleMovingAverage
from quantopian.pipeline.data.factset.estimates import PeriodicConsensus
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.experimental import risk_loading_pipeline
import quantopian.pipeline.data.factset.estimates as fe
from quantopian.pipeline.data.psychsignal import stocktwits
from quantopian.pipeline.factors import Returns
from quantopian.pipeline.data import Fundamentals

# Constraint Parameters
MAX_GROSS_LEVERAGE = 1.0
TOTAL_POSITIONS = 400

# Here we define the maximum position size that can be held for any
# given stock. If you have a different idea of what these maximum
# sizes should be, feel free to change them. Keep in mind that the
# optimizer needs some leeway in order to operate. Namely, if your
# maximum is too small, the optimizer may be overly-constrained.
MAX_SHORT_POSITION_SIZE = 1.0 / TOTAL_POSITIONS
MAX_LONG_POSITION_SIZE = 1.0 / TOTAL_POSITIONS


def initialize(context):
    """
    A core function called automatically once at the beginning of a backtest.

    Use this function for initializing state or other bookkeeping.

    Parameters
    ----------
    context : AlgorithmContext
        An object that can be used to store state that you want to maintain in
        your algorithm. context is automatically passed to initialize,
        before_trading_start, handle_data, and any functions run via schedule_function.
        context provides the portfolio attribute, which can be used to retrieve information
        about current positions.
    """

    algo.attach_pipeline(make_pipeline(), 'long_short_equity_template')

    # Attach the pipeline for the risk model factors that we
    # want to neutralize in the optimization step. The 'risk_factors' string is
    # used to retrieve the output of the pipeline in before_trading_start below.
    algo.attach_pipeline(risk_loading_pipeline(), 'risk_factors')

    # Schedule our rebalance function
    algo.schedule_function(func=rebalance,
                           date_rule=algo.date_rules.week_start(),
                           time_rule=algo.time_rules.market_open(hours=0, minutes=30),
                           half_days=True)

    # Record our portfolio variables at the end of day
    algo.schedule_function(func=record_vars,
                           date_rule=algo.date_rules.every_day(),
                           time_rule=algo.time_rules.market_close(),
                           half_days=True)

def growth ():
    qtu = QTradableStocksUS()

    fq0_eps_cons = fe.PeriodicConsensus.slice('EPS', 'qf', 0)
    fq0_eps_act = fe.Actuals.slice('EPS', 'qf', 0)

    # Get the latest mean consensus EPS estimate for the last reported quarter.
    fq0_eps_cons_mean = fq0_eps_cons.mean.latest

    # Get the EPS value from the last reported quarter.
    fq0_eps_act_value = fq0_eps_act.actual_value.latest

    # Define a surprise factor to be the relative difference between the estimated and
    # reported EPS.
    fq0_surprise = (fq0_eps_act_value - fq0_eps_cons_mean) / fq0_eps_cons_mean
    fq0_surprise_winsorize = fq0_surprise.winsorize(min_percentile=0.05, max_percentile=0.95)


    fq0_eps_cons = PeriodicConsensus.slice('EPS', 'qf', 0)
    fq1_eps_cons = PeriodicConsensus.slice('EPS', 'qf', 1)
    fq2_eps_cons = PeriodicConsensus.slice('EPS', 'qf', 2)

    #fx=pd.Series(fq2_eps_cons)

    # Each fiscal calendar has BoundColumn attributes, just like other Pipeline
    # DataSets. In this case, the 'mean' column is accessed to reference two timeseries
    # representing the mean consensus EPS estimate for the next quarter out
    # (fq1_eps_mean) and two quarters out (fq2_eps_mean). These are technically
    # pipeline factors.
    fq0_eps_mean = fq0_eps_cons.mean.latest
    fq1_eps_mean = fq1_eps_cons.mean.latest
    fq2_eps_mean = fq2_eps_cons.mean.latest

    #print type(fq2_eps_mean)

    # The above pipeline factors are used to define an estimated_growth_factor. This
    # factor looks at the relative difference each day between the mean EPS estimate
    # for two quarters out and the mean EPS estimate for next quarter.
    estimated_growth_factor1 = (fq1_eps_mean - fq0_eps_mean) / fq0_eps_mean
    estimated_growth_factor1_winsorize= estimated_growth_factor1.winsorize(min_percentile=0.05, max_percentile=0.95)

    estimated_growth_factor2 = (fq2_eps_mean - fq0_eps_mean) / fq0_eps_mean
    estimated_growth_factor2_winsorize= estimated_growth_factor2.winsorize(min_percentile=0.05, max_percentile=0.95)




    estimated_growth_12 = estimated_growth_factor1_winsorize + estimated_growth_factor2_winsorize
    estimated_growth = estimated_growth_12.rank().zscore()

    suprise = fq0_surprise_winsorize.rank().zscore()

    factor = estimated_growth + suprise

    screen = qtu & ~factor.isnull() & factor.isfinite()

    return  factor, screen


def up_and_down():
    qtu = QTradableStocksUS()

    fq1_eps_cons = fe.PeriodicConsensus.slice('EPS', 'qf', 1)
    fq1_eps_cons_up = fq1_eps_cons.up.latest
    fq1_eps_cons_down = fq1_eps_cons.down.latest

    alpha_factor = fq1_eps_cons_up - fq1_eps_cons_down
    alpha_winsorized = alpha_factor.winsorize(min_percentile=0.01,
                                              max_percentile=0.99)

    alpha_rank = alpha_winsorized.rank().zscore()

    screen = qtu & ~alpha_factor.isnull() & alpha_factor.isfinite()

    return alpha_rank, screen


def make_pipeline():
    """
    A function that creates and returns our pipeline.

    We break this piece of logic out into its own function to make it easier to
    test and modify in isolation. In particular, this function can be
    copy/pasted into research and run by itself.

    Returns
    -------
    pipe : Pipeline
        Represents computation we would like to perform on the assets that make
        it through the pipeline screen.
    """
    # The factors we create here are based on fundamentals data and a moving
    # average of sentiment data
    """
    value = Fundamentals.ebit.latest / Fundamentals.enterprise_value.latest
    quality = Fundamentals.roe.latest
    sentiment_score = SimpleMovingAverage(
        inputs=[stocktwits.bull_minus_bear],
        window_length=3,
    )

    universe = QTradableStocksUS()

    # We winsorize our factor values in order to lessen the impact of outliers
    # For more information on winsorization, please see
    # https://en.wikipedia.org/wiki/Winsorizing
    value_winsorized = value.winsorize(min_percentile=0.05, max_percentile=0.95)
    quality_winsorized = quality.winsorize(min_percentile=0.05, max_percentile=0.95)
    sentiment_score_winsorized = sentiment_score.winsorize(
        min_percentile=0.05,
        max_percentile=0.95
        )

    # Here we combine our winsorized factors, z-scoring them to equalize their influence
    combined_factor = (
        value_winsorized.zscore() +
        quality_winsorized.zscore() +
        sentiment_score_winsorized.zscore()
    )
    """
    # Build Filters representing the top and bottom baskets of stocks by our
    # combined ranking system. We'll use these as our tradeable universe each
    # day.
    qtu = QTradableStocksUS()
    up_and_down_factor, screen_up_and_down = up_and_down()
    factor_growth, screen_growth = growth()
    # Create a Returns factor with a 5-day lookback window for all securities
    # in our QTradableStocksUS Filter.
    recent_returns = Returns(window_length=5, mask=qtu)

    # Turn our recent_returns factor into a z-score factor to normalize the results.
    recent_returns_zscore = recent_returns.zscore()



    # Define high and low returns filters to be the bottom 10% and top 10% of
    # securities in the QTradableStocksUS.
    low_returns = recent_returns_zscore.percentile_between(0,10)
    high_returns = recent_returns_zscore.percentile_between(90,100)

    factor_last = 2*factor_growth + recent_returns_zscore + up_and_down_factor

    # Add a filter to the pipeline such that only high-return and low-return
    # securities are kept.
    securities_to_trade = (low_returns | high_returns)

    # Create a pipeline object to computes the recent_returns_zscore for securities
    # in the top 10% and bottom 10% (ranked by recent_returns_zscore) every day.
    pipe = Pipeline(
        columns={
            'Factor': factor_last
        },
        screen=screen_growth & screen_up_and_down
    )

    return pipe


def before_trading_start(context, data):
    """
    Optional core function called automatically before the open of each market day.

    Parameters
    ----------
    context : AlgorithmContext
        See description above.
    data : BarData
        An object that provides methods to get price and volume data, check
        whether a security exists, and check the last time a security traded.
    """
    # Call algo.pipeline_output to get the output
    # Note: this is a dataframe where the index is the SIDs for all
    # securities to pass my screen and the columns are the factors
    # added to the pipeline object above
    context.pipeline_data = algo.pipeline_output('long_short_equity_template')
    context.pipeline_data_clean = context.pipeline_data.dropna()
    # This dataframe will contain all of our risk loadings
    context.risk_loadings = algo.pipeline_output('risk_factors')


def record_vars(context, data):
    """
    A function scheduled to run every day at market close in order to record
    strategy information.

    Parameters
    ----------
    context : AlgorithmContext
        See description above.
    data : BarData
        See description above.
    """
    # Plot the number of positions over time.
    algo.record(num_positions=len(context.portfolio.positions))


# Called at the start of every month in order to rebalance
# the longs and shorts lists
def rebalance(context, data):
    """
    A function scheduled to run once every Monday at 10AM ET in order to
    rebalance the longs and shorts lists.

    Parameters
    ----------
    context : AlgorithmContext
        See description above.
    data : BarData
        See description above.
    """
    # Retrieve pipeline output
    pipeline_data = context.pipeline_data_clean

    risk_loadings = context.risk_loadings

    # Here we define our objective for the Optimize API. We have
    # selected MaximizeAlpha because we believe our combined factor
    # ranking to be proportional to expected returns. This routine
    # will optimize the expected return of our algorithm, going
    # long on the highest expected return and short on the lowest.
    objective = opt.MaximizeAlpha(pipeline_data.Factor)

    # Define the list of constraints
    constraints = []
    # Constrain our maximum gross leverage
    constraints.append(opt.MaxGrossExposure(MAX_GROSS_LEVERAGE))

    # Require our algorithm to remain dollar neutral
    constraints.append(opt.DollarNeutral())

    # Add the RiskModelExposure constraint to make use of the
    # default risk model constraints
    neutralize_risk_factors = opt.experimental.RiskModelExposure(
        risk_model_loadings=risk_loadings,
        version=0
    )
    constraints.append(neutralize_risk_factors)
    constraints.append(opt.MaxTurnover(0.2))
    # With this constraint we enforce that no position can make up
    # greater than MAX_SHORT_POSITION_SIZE on the short side and
    # no greater than MAX_LONG_POSITION_SIZE on the long side. This
    # ensures that we do not overly concentrate our portfolio in
    # one security or a small subset of securities.
    constraints.append(
        opt.PositionConcentration.with_equal_bounds(
            min=-MAX_SHORT_POSITION_SIZE,
            max=MAX_LONG_POSITION_SIZE
        ))

    # Put together all the pieces we defined above by passing
    # them into the algo.order_optimal_portfolio function. This handles
    # all of our ordering logic, assigning appropriate weights
    # to the securities in our universe to maximize our alpha with
    # respect to the given constraints.
    algo.order_optimal_portfolio(
        objective=objective,
        constraints=constraints
    )
