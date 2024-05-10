import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def plot_passengers(serie: pd.Series):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=serie.index.values,
            y=serie.values,
            mode='lines',
            name='Passengers',
            marker_color='rgb(56,41,131)',
        )
    )
    fig.add_trace(
        go.Scatter(
            x=serie.index.values,
            y=serie.rolling(12).mean().values,
            name='mean',
            marker_color='rgb(196,166,44)',
        )
    )
    fig.add_trace(
        go.Scatter(
            x=serie.index.values,
            y=serie.rolling(12).std().values,
            name='std',
            marker_color='rgb(144,12,63)',
        )
    )
    fig.update_xaxes(title="years")
    fig.update_yaxes(title="# passengers")
    fig.update_layout(
        title='# Passengers, mean and std',
        title_x=0.5,
        title_y=0.90,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    fig.show()


def plot_decomposition(trend: pd.Series, seasonality: pd.Series, residuals: pd.Series):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=trend.index.values,
            y=trend.values,
            mode='lines',
            name='Trend',
            marker_color='rgb(196,166,44)',
        )
    )
    fig.add_trace(
        go.Scatter(
            x=seasonality.index.values,
            y=seasonality.values,
            mode='lines',
            name='Seasonality',
            marker_color='rgb(56,41,131)',
        )
    )
    fig.add_trace(
        go.Scatter(
            x=residuals.index.values,
            y=residuals.values,
            mode='lines',
            name='Residuals',
            marker_color='rgb(144,12,63)',
        )
    )

    fig.update_layout(
        title='Tendency, seasonality and residuals',
        title_x=0.5,
        title_y=0.90,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    fig.show()


def plot_forecasting(
    passengers: pd.Series, model_fitted: pd.Series, predictions: pd.Series, title: str, is_log: bool, save: bool = False
):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=passengers.index.values,
            y=passengers.values,
            mode='lines',
            name='Original',
            marker_color='rgb(196,166,44)',
            opacity=0.6,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=model_fitted.fittedvalues.index.values[2:],
            y=model_fitted.fittedvalues.values[2:],
            mode='lines',
            name='Fitted Train',
            marker_color='rgb(56,41,131)',
            opacity=0.6,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=predictions.index.values,
            y=predictions.values,
            mode='lines',
            name='Predictions',
            marker_color='rgb(180,230,122)',
        )
    )
    fig.update_xaxes(title="years")
    fig.update_yaxes(title="# passengers")
    if is_log:
        range = [4, 7]
        y_title = 'log(# passengers)'
    else:
        range = [50, 1500]
        y_title = '# passenger'
    fig.update_layout(
        yaxis=dict(title=y_title, range=range),  # Set y-axis limits (ylim) from 0 to 80
        title=title,
        title_x=0.5,
        title_y=0.90,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=1,
            xanchor="right",
        ),
    )

    if save:
        fig.write_image(f'images/result_forecasting/forecast_{len(predictions)}_months.png')
    fig.show()


def get_metrics(model_fit, predictions, train_set, test_set, name):
    R2 = r2_score(train_set[2:], model_fit.fittedvalues[2:])
    mse_e = mean_squared_error(train_set[2:], model_fit.fittedvalues[2:])  # 2: due to differencing
    mae_e = mean_absolute_error(train_set[2:], model_fit.fittedvalues[2:])

    # Test
    R2_test = r2_score(test_set, predictions)
    mse_e_test = mean_squared_error(test_set, predictions)
    mae_e_test = mean_absolute_error(test_set, predictions)

    print(
        pd.DataFrame(
            {
                'Model': [name, name],
                'Mode': ['Training', 'Testing'],
                'R2': [R2, R2_test],
                'MSE': [mse_e, mse_e_test],
                'MAE': [mae_e, mae_e_test],
            }
        )
    )
