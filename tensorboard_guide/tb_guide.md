# Train Python Code

* import the `TensorBoardLogger`:

```python
from pytorch_lightning.loggers import TensorBoardLogger
```

* in the `shared_epoch_end` function of the model, log the parameters, in addition to the `step` parameter:

```python
class TestModel(pl.LightningModule):
  ...

    def shared_epoch_end(self, outputs, stage):
        ...
        metrics = {...} # dictionary
        self.log_dict({**metrics,"step": self.current_epoch}, prog_bar=True)

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")
```

* in the `main` function (or the `if __name__=='__main__'`) add the `tensorboard_logs_path` argument to the parse:
```python
def main():
    parser = argparse.ArgumentParser()
    ...
    parser.add_argument("--tensorboard_logs_path", type=str, required=True, help="Path used for writing TensorFlow logs")

    train(parser.parse_args())
```

* in the `train` function, add the `TensorBoardLogger` to the trainer:
```python
def train(args):
    ...

    logs_dir = args.tensorboard_logs_path
    logger = TensorBoardLogger(save_dir=logs_dir)
    
    trainer = pl.Trainer(
        ...,
        logger=logger,
        ...
    )
```

# Estimator Notebook in Sagemaker

* import the `TensorBoardOutputConfig`:
```python
from sagemaker.debugger import TensorBoardOutputConfig
```

* setup the local log dir and the s3 output for the tensorboard logs:
```pyton
LOG_DIR="/opt/ml/output/tensorboard"
tensorboard_logs_path = "s3://bucket/path/to/tensorboard_logs"
```

* define the `tensorboard_output_config`:
```python
tensorboard_output_config = TensorBoardOutputConfig(
    s3_output_path=tensorboard_logs_path,
    container_local_output_path=LOG_DIR
)
```

* add the `tensorboard_output_config` argument to the estimator, and the `tensorboard_logs_path` hyperparameter:
```python
estimator = PyTorch(
      ...,
      tensorboard_output_config=tensorboard_output_config,
      ...
      hyperparameters={
          ...,
          'tensorboard_logs_path': LOG_DIR,
          ...
      }
)
```

## Running the Estimator

now we are ready to execute
```python
estimator.fit({"training": training_dataset_s3_path}, logs=True, wait=False)
```


# TensorBoard

We can access tensorboard with the following command in our estimator notebook:
```python
from sagemaker.interactive_apps import tensorboard

region = "eu-west-1"
app = tensorboard.TensorBoardApp(region)

print("Navigate to the following URL:")
print(app.get_app_url(training_job_name=estimator.latest_training_job.name))
```
which will print a url like so:

```
>>> Navigate to the following URL:

>>> https://<id>.studio.<region>.sagemaker.aws/tensorboard/default/data/plugin/sagemaker_data_manager/add_folder_or_job?Redirect=True&Name=<training_job_name>
```

and then enter the link and continue from step 6, or we can access it from sageamker and start from step 1:

1. search for `amazon sagemaker` and click it

![step 1](https://raw.githubusercontent.com/taloy42/TalPosts/main/tensorboard_guide/1.png)

2. click on your domain, in our case `sarine-...`

![step 2](https://raw.githubusercontent.com/taloy42/TalPosts/main/tensorboard_guide/2.png)

3. click on your username, `tallevy` in my case

![step 3](https://raw.githubusercontent.com/taloy42/TalPosts/main/tensorboard_guide/3.png)

4. click `Launch`

![step 4](https://raw.githubusercontent.com/taloy42/TalPosts/main/tensorboard_guide/4.png)

5. click on `TensorBoard`

![step 5](https://raw.githubusercontent.com/taloy42/TalPosts/main/tensorboard_guide/5.png)

6. now we can see all of the training jobs that were run with `Tensorboard`.

![step 6](https://raw.githubusercontent.com/taloy42/TalPosts/main/tensorboard_guide/6.png)

7. We can click on the info button next to the training job's name to see more info about it

![step 7](https://raw.githubusercontent.com/taloy42/TalPosts/main/tensorboard_guide/7.png)

8. for example we can see the number of epochs (and all of the other hyperparameters). we can now close the info

![step 8](https://raw.githubusercontent.com/taloy42/TalPosts/main/tensorboard_guide/8.png)

9. we can click on the checkbox next to the training job to select it (we can choose multiple) and then select `Add Selected Jobs`

![step 9](https://raw.githubusercontent.com/taloy42/TalPosts/main/tensorboard_guide/9.png)

10. now two new tabs have opened, `Time Series` and `Scalars`. we can monitor our logged valued using both. in the next step we clicked on `Time Series`

![step 10](https://raw.githubusercontent.com/taloy42/TalPosts/main/tensorboard_guide/10.png)

11. we can see the graph with `epochs` as the X-axis (listed as `step`). in each graph there are 2 plots, a bold one and a faded one. the faded one is the raw values, and the bold one is the smoothed values, by the [exponential smoothing](https://en.wikipedia.org/wiki/Exponential_smoothing) function, with $\alpha$ set to the `Smoothing` parameter to the left.

![step 11](https://raw.githubusercontent.com/taloy42/TalPosts/main/tensorboard_guide/11.png)


