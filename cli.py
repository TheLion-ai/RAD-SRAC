import click
from typing_extensions import Annotated

from src.config.datasets import datasets
from src.config.models import models
from src.config.encoder import encoders
from src.analyze_dataset import DatasetAnalyzer
from src.vector_db.retriever import Retreiver


# Create dynamic string-based Enum classes based on configuration


@click.command()
@click.option(
    "--dataset",
    prompt="Select dataset",
    help="Name od the dataset",
    type=click.Choice(datasets.keys()),
)
@click.option(
    "--model",
    prompt="Select model",
    help="Name od the model",
    type=click.Choice(models.keys()),
)
@click.option(
    "--encoder",
    prompt="Select encoder",
    help="Name od the encoder",
    type=click.Choice(encoders.keys()),
)
@click.option("--few-shot/--no-few-shot", default=False, prompt="Run few shot?")
@click.option(
    "--n-images",
    prompt="No. of images",
    help="How many relevant images to show?",
    type=int,
    default=5,
)
@click.option(
    "--classes/--no-classes",
    default=True,
    prompt="Add true classes in few-shot prompt?",
)
@click.option(
    "--roco", is_flag=True, help="Use ROCO database instead of the dataset database"
)
@click.option("--verbose", is_flag=True, help="Log each prediction")
def run(
    dataset: str,
    model: str,
    encoder: str,
    few_shot: bool,
    n_images: int,
    classes: bool,
    roco: bool,
    verbose: bool,
):
    """
    Analyze medical images using the specified dataset, model, and encoder.
    """
    dataset = datasets[dataset]
    model = models[model]
    encoder = encoders[encoder]
    collection_name = "ROCO" if roco else dataset.name
    retriever = Retreiver(encoder, collection_name=collection_name, n_images=n_images)
    print(f"classer {classes}")
    dataset_analyzer = DatasetAnalyzer(
        dataset, model, few_shot, retriever, verbose, classes, n_images
    )
    dataset_analyzer.analyze_imgs()


@click.command()
@click.option(
    "--dataset",
    prompt="Select dataset",
    help="Name od the dataset",
    type=click.Choice(datasets.keys()),
)
@click.option(
    "--encoder",
    prompt="Select encoeder",
    help="Name od the encoder",
    type=click.Choice(encoders.keys()),
)
def upload_dataset(dataset: str, encoder: str):
    dataset = datasets[dataset]
    encoder = encoders[encoder]
    retreiver = Retreiver(encoder=encoder, collection_name=dataset.name)
    retreiver.upload_dataset(dataset.train_file)


if __name__ == "__main__":

    @click.group()
    def cli() -> None:
        pass

    cli.add_command(run)
    cli.add_command(upload_dataset)

    cli()
    upload_dataset()

    # run()
