import matplotlib.ticker as ticker
import datetime


def change_xaxis_to_date(axis, ref_date):
    """
    Change the format of a numerically formatted x-axis to date.
    """

    def to_date(x_value, pos):
        date = ref_date + datetime.timedelta(days=x_value)
        return date.strftime("%d-%m")

    date_format = ticker.FuncFormatter(to_date)
    axis.xaxis.set_major_formatter(date_format)
    axis.xaxis.set_tick_params(rotation=30)
