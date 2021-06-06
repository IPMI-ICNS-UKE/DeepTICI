#DeepTICI

Implementation of the first fully automatic TICI-scoring system. The system is based on Combination of *encoder*, *GRU*
and *classifier*. Details in regard to the Method can be found in the following MICCAI contribution:
[Time Matters: Handling Spatio-Temporal Perfusion Information for Automated TICI Scoring](https://doi.org/10.1007/978-3-030-59725-2_9)
![alt text](images/Method.png "Method overview")

A detailed performance analysis is available in Stroke:
[Deep learning-based automated TICI scoring:a timely proof-of-principle study ](working_doi)
![alt text](images/Results.jpg "Results overview")

Example usage for prediction after installation. For predicting purposes weights from all experiments are included:

```python
from DeepTICI import predict


dcm_paths = ['/some/path/to/view/1', '/some/path/to/view/2']
tici_score = predict.predict_series(dcm_paths)
print(tici_score)
```

Furthermore fine tuning on personal data is supported:

```python
from DeepTICI import model
from DeepTICI.helper import ModelMode, OutputMode

#model init
model = model.TICIModelHandler(num_classes=5, feature_size= 1280, in_channels=3)

#forward pass
for data in data_loader:
    # DCM-series with shape: batch x time x 2 (views) x height x width
    img = data[0]
    series_lenghts = data[1]
    output = model(x=img, series_lenghts=series_lenghts, model_mode=ModelMode.train, output_mode=OutputMode.last_frame)
    # loss, backward-pass,  etc.
```
