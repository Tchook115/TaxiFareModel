from sklearn.preprocessing import OneHotEncoder
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data
from sklearn.model_selection import train_test_split
from memoized_property import memoized_property
import mlflow
from mlflow.tracking import MlflowClient
import joblib
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

MLFLOW_URI = "https://mlflow.lewagon.co/"
EXPERIMENT_NAME = "FR Bordeax Tchook115 TaxiFareModel + 0.0.1"

class Trainer():
    def __init__(self, X, y, kwargs):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = EXPERIMENT_NAME
        self.model = kwargs['model']
        self.distance = kwargs['distance']


    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(
                self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


    def set_pipeline(self):
        '''returns a pipelined model'''
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        self.pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('model', self.model)
        ])


    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipe.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        '''returns the value of the RMSE'''
        y_pred = self.pipe.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse

    def save_model(self):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self.pipe, 'model.joblib')


if __name__ == "__main__":

    params1 = {'model': Ridge(), 'distance':'Manhattan', 'n_rows':10000}
    params2 = {'model': Ridge(), 'distance':'Manhattan', 'n_rows':1000000}
    params3 = {'model': LinearRegression(), 'distance':'Manhattan', 'n_rows':10000}
    params4 = {'model': LinearRegression(),'distance': 'Manhattan','n_rows': 1000000}
    params5 = {
        'model': RandomForestRegressor(),
        'distance': 'Manhattan',
        'n_rows': 10000
    }
    params6 = {
    'model': RandomForestRegressor(),
    'distance': 'Manhattan',
    'n_rows': 10000
    }

    param_lst = [params1, params2, params3, params4, params5, params6]


    for param in param_lst:
        # get data
        df = get_data(nrows=param['n_rows'])
        # clean data
        df = clean_data(df)
        # set X and y
        y = df["fare_amount"]
        X = df.drop("fare_amount", axis=1)
        # hold out
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
        # train
        train = Trainer(X_train, y_train, param)
        train.run()
        # evaluate
        rmse = train.evaluate(X_val, y_val)
        train.mlflow_log_metric('rmse', rmse)
        train.save_model()
        train.mlflow_log_param('n_row model', (param['model'],param['n_rows']))
        print(rmse)
