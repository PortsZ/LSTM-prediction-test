from datetime import datetime
import backtrader as bt
import backtrader.indicators as btind



# Define your strategy
class MeanReversionStrategy(bt.Strategy):
    params = (
        ('ma_short_period', 50),
        ('ma_medium_period', 100),
        ('ma_long_period', 200),
        ('volume_lookback_short', 10),
        ('volume_lookback_long', 90),
        ('volume_multiplier', 1),
         ('exit_days', 90),  # Number of days to hold before exiting
         ('printlog', False),
    )

    def log(self, txt, dt=None):
        ''' Logging function '''
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()} {txt}')

    def __init__(self):
        # Moving Averages
        self.ma_short = btind.MovingAverageSimple(self.data.close, period=self.params.ma_short_period)
        self.ma_medium = btind.MovingAverageSimple(self.data.close, period=self.params.ma_medium_period)
        self.ma_long = btind.MovingAverageSimple(self.data.close, period=self.params.ma_long_period)

        # Volume
        self.volume_avg_short = btind.MovingAverageSimple(self.data.volume, period=self.params.volume_lookback_short)
        self.volume_avg_long = btind.MovingAverageSimple(self.data.volume, period=self.params.volume_lookback_long)

        # Crossover Indicators
        self.crossover_up = btind.CrossUp(self.ma_short, self.ma_medium)
        
        self.order = None
        self.entry_day = None

    def next(self):
        current_day = len(self)  # Current day index

        # Check if an order is pending
        if self.order:
            return

        # Check if we are in the market
        if not self.position:
            # Entry logic
            if self.volume_avg_short[0] > self.params.volume_multiplier * self.volume_avg_long[0] and \
               self.ma_short[0] > self.ma_short[-1] and self.ma_medium[0] > self.ma_medium[-1]:

                self.log('BUY CREATE, %.2f' % self.data.close[0])
                self.order = self.buy()
                self.entry_day = current_day  # Store the day of entry
        else:
            if self.ma_long[0] < self.ma_long[-1]:
                self.log('SELL CREATE, %.2f' % self.data.close[0])
                self.order = self.sell()
            # if current_day - self.entry_day >= self.params.exit_days:
            #     self.log('SELL CREATE, %.2f' % self.data.close[0])
            #     self.order = self.sell()

    def notify_order(self, order):
        # Reset order to None once it is completed
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None

# Create a Cerebro engine
cerebro = bt.Cerebro()

# Add strategy
cerebro.addstrategy(MeanReversionStrategy)

# Load data (modify path and format as needed)
data = bt.feeds.GenericCSVData(
    dataname='./data/SPY.csv',
    fromdate=datetime(1993, 1, 29),
    todate=datetime(2023, 11, 9),
    nullvalue=0.0,
    dtformat=('%Y-%m-%d'),
    datetime=0,
    high=2,
    low=3,
    open=1,
    close=4,
    volume=6,
    openinterest=-1
)

# Add the data to Cerebro
cerebro.adddata(data)

# Set broker parameters
cerebro.broker.set_cash(1000000)
cerebro.broker.setcommission(commission=0.001)

# Run the strategy
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
cerebro.run()
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
