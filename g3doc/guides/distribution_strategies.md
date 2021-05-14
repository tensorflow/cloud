# Cluster and distribution strategy configuration

By default, `run` API takes care of wrapping your model code in a TensorFlow
distribution strategy based on the cluster configuration you have provided.

## No distribution

CPU chief config and no additional workers

```python
tfc.run(entry_point='mnist_example.py',
        chief_config=tfc.COMMON_MACHINE_CONFIGS['CPU'])
```

## `OneDeviceStrategy`

1 GPU on chief (defaults to `AcceleratorType.NVIDIA_TESLA_T4`) and no additional
workers.

```python
tfc.run(entry_point='mnist_example.py')
```

## `MirroredStrategy`

Chief config with multiple GPUs (`AcceleratorType.NVIDIA_TESLA_V100`).

```python
tfc.run(entry_point='mnist_example.py',
        chief_config=tfc.COMMON_MACHINE_CONFIGS['V100_4X'])
```

## `MultiWorkerMirroredStrategy`

Chief config with 1 GPU and 2 workers each with 8 GPUs
(`AcceleratorType.NVIDIA_TESLA_V100`).

```python
tfc.run(entry_point='mnist_example.py',
        chief_config=tfc.COMMON_MACHINE_CONFIGS['V100_1X'],
        worker_count=2,
        worker_config=tfc.COMMON_MACHINE_CONFIGS['V100_8X'])
```

## `TPUStrategy`

Chief config with 1 CPU and 1 worker with TPU.

```python
tfc.run(entry_point="mnist_example.py",
        chief_config=tfc.COMMON_MACHINE_CONFIGS["CPU"],
        worker_count=1,
        worker_config=tfc.COMMON_MACHINE_CONFIGS["TPU"])
```

Please note that TPUStrategy with TensorFlow Cloud works only with TF version
2.1 as this is the latest version supported by
[AI Platform cloud TPU](https://cloud.google.com/ai-platform/training/docs/runtime-version-list#tpu-support)

## Custom distribution strategy

If you would like to take care of specifying a distribution strategy in your
model code and do not want `run` API to create a strategy, then set
`distribution_stategy` as `None`. This will be required, for example, when you
are using `strategy.experimental_distribute_dataset`.

```python
tfc.run(entry_point='mnist_example.py',
        distribution_strategy=None,
        worker_count=2)
```
