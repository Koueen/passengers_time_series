import json
import pickle
from datetime import datetime
from typing import List, Union

import numpy as np
import pandas as pd

from analysis.utils import plot_forecasting


class NPassengersForecast:
    """Class used to forecast the number of passengers per month"""

    def __init__(self, model_dir: str = 'model/sarima.pkl', data_dir: pd.DataFrame = 'data/flights.csv'):
        """Initialize NPassengersForecast

        Args:
            model_dir (str): Path to model (HoltWinters).
            data_dir (str): Path to data used for training.
        """
        self.models_dir = model_dir
        self.data_dir = data_dir
        self.model = self.load_model()

    def __call__(self, n_months: int) -> pd.Series:
        """Predict the next n_months number of passengers

        Args:
            n_months (int): Number of months to forecast

        Returns:
            pd.Series: Index DateTime and passengers (float)
        """
        df = self.preprocess()  # This is used to plot to know from which point the forecasting is being predicted
        predictions = self.predict(n_months)
        plot_forecasting(df.passengers, self.model, predictions, 'Damped HoltWinters', False, True)
        return predictions

    def preprocess(self) -> pd.DataFrame:
        """Preprocess raw csv into clean dataframe

        Returns:
            pd.DataFrame: Composed by year | month | passengers. Their index is a datetime "year-month-day"
        """
        df = pd.read_csv(self.data_dir)
        df['month'] = df['month'].apply(lambda x: datetime.strptime(x, '%B').month)
        df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
        df.set_index('date', inplace=True)
        df = df.sort_values(by='date')
        return df

    def load_model(self):
        """Load model

        Returns:
            HoltWinters model
        """
        return pickle.load(open(self.models_dir, 'rb'))

    def predict(self, n_months: int) -> pd.Series:
        """Predict n_months

        Args:
            n_months: Number of months to predict in the future

        Returns:
            Returns:
            pd.Series: Index DateTime and passengers (float)
        """
        log_preds = self.model.forecast(n_months)
        predictions = round(np.exp(log_preds))
        return predictions


if __name__ == '__main__':
    passgenerf = NPassengersForecast()
    prediction = passgenerf(100)
    print(prediction)
