import bentoml
import pandas as pd

model_runner = bentoml.mlflow.import_model("model", model_uri="../git models/tfidf_logit_model",
                                           signatures={'predict': {'batchable': True}}
                                           ).to_runner()
input_spec = bentoml.io.PandasDataFrame.from_sample(pd.DataFrame(['This is a sample text'],
                                                                 columns=['tweet_text']),
                                                    enforce_shape=False)

svc = bentoml.Service('mlflow_cyberbullying', runners=[model_runner])


@svc.api(input=input_spec, output=bentoml.io.NumpyNdarray())
def predict(input_arr):
    return model_runner.predict.run(input_arr)
