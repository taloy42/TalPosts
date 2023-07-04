# In the Notebook
## Optional - Copy Checkpoint from another project
add the following function to the notebook:
```python
def copy_checkpoint(from_bucket, from_prefix, to_bucket, to_prefix, epoch_num):
    print(from_prefix, to_prefix)
    s3 = boto3.resource('s3')

    source_bucket = s3.Bucket(from_bucket)
    if from_bucket != to_bucket:
        dest_bucket = s3.Bucket(to_bucket)
    else:
        dest_bucket = source_bucket
    for obj in source_bucket.objects.filter(Prefix=from_prefix):
        if str(epoch_num) in obj.key:
            old_source = { 'Bucket': from_bucket,
                   'Key': obj.key}
            
            new_key = os.path.join(to_prefix,obj.key[len(from_prefix):])
            new_obj = source_bucket.Object(new_key)
            print('copied {}/{} to {}/{}'.format(from_bucket,obj.key, to_bucket, new_obj.key))
            new_obj.copy(old_source)
            return new_obj

    return None
```

for example, if the checkpoint is currently stored in `s3://old_bucket/path/to/checkpoints/epoch=69-loss=0.6.ckpt` and we want to start a training  job that saves checkpoints to `s3://new_bucekt/path/to/new/checkpoints/`, then we will run the function as:
```python
from_bucket = 'old_bucket'
from_prefix = 'path/to/checkpoints/'

to_bucket = 'new_bucket'
to_prefix = 'path/to/new/checkpoints/'

Epoch_start_num = 69

copy_checkpoint(
	from_bucket, from_prefix,
	to_bucket, to_prefix,
	Epoch_start_num
	)			
```

## Changes to the estimator
add the `Epoch_start_num` hyperparameter to the estimator:
```python
estimator = PyTorch(
	...,
	hyperparameters={
        ...,
        'Epoch_start_num' : Epoch_start_num
    }
) 
```
whetr `Epoch_start_num` is the epoch we want to start the checkpoint from

# In the .py file
## In the `if __name__=='__main__'`
add the `Epoch_start_num` hyperparameter:
```python
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	...
	parser.add_argument('--Epoch_start_num', type=int, default=-1, help='Epoch start for training')
	...
```
## Add the `get_checkpoint` function
add the following function to the code:
```python
import os
import time

def get_checkpoint(args, verbose=True):
    checkpoints = os.listdir(args.checkpoint_path)
    checkpoint_options = []
    for c in checkpoints:
        if f'epoch={args.Epoch_start_num:02d}' in c:
            checkpoint_options.append(c)

    if len(checkpoint_options)<1:
        checkpoint = None
    else:
        checkpoint = max(checkpoint_options,key=lambda x:os.path.getmtime(os.path.join(args.checkpoint_path,x)))

    if verbose:
        if checkpoint is None:
            checkpoint_not_found_msg = "=======================================\ncheckpoint with epoch=`{:02d}` not found in `{}`:\n"
            print(checkpoint_not_found_msg.format(args.Epoch_start_num,args.checkpoint_path,checkpoints))
            for x in sorted(checkpoints,key=lambda x:(x[:8],os.path.getmtime(os.path.join(args.checkpoint_path,x)))):
                print(f"{x} => {time.ctime(os.path.getmtime(os.path.join(args.checkpoint_path,x)))}")
            print("=======================================")        
        else:
            checkpoint_found_msg = "=======================================\nloading from checkpoint `{}`\n======================================="
            print(checkpoint_found_msg.format(checkpoint))
    return checkpoint
```
## In the `train` function
assuming the module's name is `PLModule`:
```python
class PLModule(pl.LightningModule):
    def __init__(self, param1, param2):
        super().__init__()
        self.save_hyperparameters()
        ...
```
wrap the model instantiation:
```python
model = PLModule(param1=value1, param2=value2)
```
 with the following:
```python
if  os.path.isdir(args.checkpoint_path):
    print("Checkpointing directory {} exists".format(args.checkpoint_path))
else:
    print("Creating Checkpointing directory {}".format(args.checkpoint_path))
    os.makedirs(args.checkpoint_path,exist_ok=True)

checkpoint = get_checkpoint(args)
model_params = dict(
    param1=value1,
    param2=value2,
)
if checkpoint is None:
    model = PLModule(**model_params)
else:
    model = PLModule.load_from_checkpoint(os.path.join(args.checkpoint_path,checkpoint),**model_params)
```

the `train` function should look like this:
```python
def train(args):
    if os.path.isdir(args.checkpoint_path):
        print("Checkpointing directory {} exists".format(args.checkpoint_path))
    else:
        print("Creating Checkpointing directory {}".format(args.checkpoint_path))
        os.makedirs(args.checkpoint_path,exist_ok=True)
        
    checkpoint = get_checkpoint(args)
    model_params = dict(
        param1=value1,
        param2=value2,
    )
    if checkpoint is None:
        model = PLModule(**model_params)
    else:
        model = PLModule.load_from_checkpoint(os.path.join(args.checkpoint_path,checkpoint),**model_params)
    train_loader = DataLoader(dataset)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_path,
        filename="{epoch:02d}",
        save_top_k=-1,
        ...,
    )

    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
        ...,
    )

    trainer.fit(model=model, train_dataloaders=train_loader)
```
