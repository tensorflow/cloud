from typing import Optional
from typing import Any
from typing import Dict

try:
    import typer
except (ImportError, ModuleNotFoundError) as inst:
    print(type(inst))
    print(inst)
    print('You can try running "pip install colorama==0.4.4 typer[all]==0.3.2" on you terminal')

from src.python.tensorflow_cloud.core.run import run
from src.python.tensorflow_cloud.core.run import remote
from src.python.tensorflow_cloud.core import docker_config as docker_config_module
from src.python.tensorflow_cloud.core.machine_config import COMMON_MACHINE_CONFIGS

app = typer.Typer()

@app.command("remote")
def remote_command():
    """
    To know is you code is running remote with TF Cloud.
    """
    if remote():
        typer.echo("Running remotly")
        typer.Exit()
    
    return typer.echo("Running Localy")

@app.command("run", help="Run your code in a remote cloud environment with TF Cloud.")
def run_command(
    entry_point: Optional[str] = typer.Argument(..., help="File path to the python file or iPython notebook that contains the TensorFlow code"),
    requirements_txt: Optional[str] = typer.Option(None, help="File path to requirements.txt file containing additional pip dependencies if any"),
    image_uri: Optional[str] = typer.Option(None, help="Docker image URI for the Docker image being built"),
    parent_image: Optional[str] = typer.Option(None, help="Parent Docker image to use. Example value - 'gcr.io/my_gcp_project/deep_learning:v2' If a parent Docker image is not provided here, we will use a [TensorFlow Docker image](https://www.tensorflow.org/install/docker) as the parent image."),
    cache_from: Optional[str] = typer.Option(None, help="Docker image URI to be used as a cache when building the new Docker image. This is especially useful if you are iteratively improving your model architecture/training code. If this parameter is not provided, then we will use `image` URI as cache."),
    image_build_bucket: Optional[str] = typer.Option(None, help="GCS bucket name to be used for building a Docker image via [Google Cloud Build](https://cloud.google.com/cloud-build/). If it is not specified, then your local Docker daemon will be used for Docker build."),
    distribution_strategy: str = typer.Option("auto", help="Tensorflow distribution strategy based on the machine config"),
    chief_config: str = typer.Option("auto", help="`MachineConfig` that represents the configuration for the chief worker in a distribution cluster. Choose between (CPU, K80_1X, K80_4X, K80_8X, P100_1X, P100_4X, P4_1X, P4_4X, V100_1X, V100_4X, T4_1X, T4_4X, TPU)"),
    worker_config: str = typer.Option("auto", help="`MachineConfig` that represents the configuration for the general workers in a distribution cluster. Choose between (CPU, K80_1X, K80_4X, K80_8X, P100_1X, P100_4X, P4_1X, P4_4X, V100_1X, V100_4X, T4_1X, T4_4X, TPU)"),
    worker_count: int = typer.Option(0, help="Represents the number of general workers in a distribution cluster."),
    entry_point_args: str = typer.Option(None, help="Command line arguments to pass to the `entry_point` program. Not implemented yet."), # review
    stream_logs: bool = typer.Option(False, help="Boolean flag which when enabled streams logs back from the cloud job."),
    job_labels: str = typer.Option(None, help="Labels to organize jobs. You can specify up to 64 key-value pairs in lowercase letters and numbers, where the first character must be lowercase letter. For more details see https://cloud.google.com/ai-platform/training/docs/resource-labels. Not implemented yet.") # review
):
    entry_point_args = None
    job_labels = None
    
    docker_config = docker_config_module.DockerConfig(image=image_uri, 
                                                    parent_image=parent_image,
                                                    cache_from=cache_from,
                                                    image_build_bucket=image_build_bucket)
    
    if chief_config != "auto":
        try:
            chief_config = COMMON_MACHINE_CONFIGS[chief_config]
        except KeyError:
            typer.BadParameter("You need to choose a between (CPU, K80_1X, K80_4X, K80_8X, P100_1X, P100_4X, P4_1X, P4_4X, V100_1X, V100_4X, T4_1X, T4_4X, TPU) options")
    
    if worker_config != "auto":
        try:
            worker_config = COMMON_MACHINE_CONFIGS[worker_config]
        except KeyError:
            typer.BadParameter("You need to choose a between (CPU, K80_1X, K80_4X, K80_8X, P100_1X, P100_4X, P4_1X, P4_4X, V100_1X, V100_4X, T4_1X, T4_4X, TPU) options")
    
    info = run(
        entry_point=entry_point,
        requirements_txt=requirements_txt,
        docker_config=docker_config,
        distribution_strategy=distribution_strategy,
        chief_config=chief_config,
        worker_config=worker_config,
        worker_count=worker_count,
        entry_point_args=entry_point_args,
        stream_logs=stream_logs,
        job_labels=job_labels
    )
    
    typer.echo(f"Job id: {info['job_id']}")
    typer.echo(f"Docker image URI: {info['docker_image']}")

if __name__ == '__main__':
    app()