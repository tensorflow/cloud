import tensorflow_cloud as tfc

tfc.run(
    entry_point="train_model.py",
    requirements_txt="requirements.txt",
    stream_logs=True,
)
