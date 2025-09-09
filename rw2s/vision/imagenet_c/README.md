Running the motivating example:
- On ImageNet-C:
```python
python rw2s/imagenet_c/run_weak_strong.py \
    --batch_size=128 \
    --weak_model_name="alexnet" \
    --strong_model_name="resnet50_dino" \
    --n_train=40000 \
    --n_epochs=20 \
    --lr=1e-3 \
    --seed=0 \
    --data_path="/mlbio_scratch/sobotka/data/imagenet_c" \
    --save_path="/mlbio_scratch/sobotka/w2s/imagenet_c" \
    --data_name="gaussian_noise/3" \
    --save_cache=False \
    --load_cache=False \
    --device=7
```


- On ImageNet:
```python
python rw2s/imagenet_c/run_weak_strong.py \
    --batch_size=128 \
    --weak_model_name="alexnet" \
    --strong_model_name="resnet50_dino" \
    --n_train=40000 \
    --n_epochs=20 \
    --lr=1e-3 \
    --seed=0 \
    --data_path="/mlbio_scratch/sobotka/data/imagenet" \
    --save_path="/mlbio_scratch/sobotka/w2s/imagenet" \
    --data_name="imagenet_val" \
    --save_cache=False \
    --load_cache=False \
    --device=6
```
