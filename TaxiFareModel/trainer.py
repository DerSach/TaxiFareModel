# imports
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.utils import haversine_vectorized, compute_rmse
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


class Trainer():
    
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
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
            ('linear_model', LinearRegression())
        ])

    def run(self):
        '''returns a trained pipelined model'''
        self.set_pipeline()
        pipe_trained = self.pipe.fit(self.X, self.y)
        return pipe_trained

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        pipe_trained = self.run()
        y_pred = pipe_trained.predict(X_test)
        rmse = compute_rmse(y_pred,y_test)
        return rmse


if __name__ == "__main__":
    n_rows = 10000
    
    df = get_data(n_rows)
    df = clean_data(df)
    
    y = df['fare_amount']
    X = df.drop(columns = 'fare_amount')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    
    trainer = Trainer(X_train, y_train)
    trainer.run()
    
    rmse = trainer.evaluate(X_test, y_test)
    
    print(rmse)
    print('OK model trained')
