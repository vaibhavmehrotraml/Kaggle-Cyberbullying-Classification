import mlflow
import logging
import pandas as pd
import argparse
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# TODO: Best logging practices

logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                    )

# Set up parser for CLI arguments
parser = argparse.ArgumentParser(
    description="Preprocess the given files using the given arguments.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-f", "--file",
                    action='store',
                    dest='filename',
                    type=str,
                    help="Define the file path of the training data.")
parser.add_argument("-ts", "--test-size",
                    action='store',
                    dest='test_size',
                    type=float,
                    default=0.2,
                    help="Define the train-test split of the training data.")

def load_data(data):
    logging.info(f'Starting data extraction from {data} file')
    df = pd.read_csv(data)
    return df


def prepare_data(df, test_size):
    logging.info(f'Preparing test-training splits from data of length {len(df)}')
    X = df.drop('cyberbullying_type', axis=1)
    y = df['cyberbullying_type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, stratify=y, random_state=42)
    return X_train, X_test, y_train, y_test


def create_model():
    transformer = Pipeline(steps=[('countVectorizer', CountVectorizer()),
                                  ('tfIdfTransformer', TfidfTransformer())])

    preprocessor = ColumnTransformer(transformers=[('transformer', transformer, 'tweet_text')
                                                   ], remainder='passthrough')

    classifier = Pipeline(steps=[('logisticRegressionClassifier', LogisticRegression(max_iter=5000))])
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', classifier)])
    return model


def train_model(model, X_train, y_train):
    with mlflow.start_run(run_name="train") as run:
        model.fit(X_train, y_train)
        mlflow.set_tag("features", str(X_train.columns.values.tolist()))
        signature = mlflow.models.infer_signature(
            model_input=X_train,
            model_output=model.predict(X_train)
        )
        mlflow.sklearn.log_model(model, "model", signature=signature)
    return model


def get_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    f1_metric = f1_score(y_test, y_pred, average='weighted')
    mlflow.log_metric("F1_Score", f1_metric)
    return f1_metric


if __name__ == '__main__':
    # Parse arguments
    args = parser.parse_args()
    filename = args.filename
    test_size = args.test_size

    df = load_data('data/cyberbullying_tweets_clean.csv')
    X_train, X_test, y_train, y_test = prepare_data(df=df, test_size=test_size)
    model = create_model()
    model = train_model(model=model, X_train=X_train, y_train=y_train)
    get_metrics(model=model, X_test=X_test, y_test=y_test)
