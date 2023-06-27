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

## In the `train` function
replace the `trainer.fit` call with the following:
```python
# check if checkpoint exists
checkpoints = os.listdir(args.checkpoint_path)
checkpoint = None
for c in checkpoints:
    if f'epoch={args.Epoch_start_num:02d}' in c:
        checkpoint = c
	    break
            
# if checkpoint exists, start from it
if checkpoint is None:
    print(f"""
============================================================================================
didn't find chekpoint {args.Epoch_start_num:02d} in the contents of {args.checkpoint_path}:
{checkpoints}
=============================================================================================
""")
	trainer.fit(
    ...
	)
else:
	print("=======================================\ntraining from checkpoint {args.Epoch_start_num:02d}\n===================================")
	trainer.fit(
		...,
		ckpt_path = os.path.join(args.checkpoint_path,checkpoint),
    )
```
