"""
python -m titanic
"""
import click
import os

from .dataset import get_input_data
from .models import MLP
from .train import trainer


@click.group()
def cli():
    pass

@cli.command()
@click.option("--epochs", type=int, default=10, help="Training total epochs")
@click.option("--batch_size", type=int, default=32, help="Training batch size")
@click.option("--lr", type=float, default=1e-3, help="Training learn rate")
@click.option("--train_file", type=str, default="train.csv", help="Training .csv file name")
@click.option("--save_dir", type=str, default="model/", help="Model saved directory")
@click.option("--test_file", type=str, default="test.csv", help="Testing .csv file name")
@click.option("--output_dir", type=str, default="output/", help="Outputed .csv saved directory")
@click.option("--output_name", type=str, default="submission.csv", help="Outputed .csv file name")
def train(epochs, batch_size, lr, train_file, save_dir,
          test_file, output_dir, output_name):
    # Train
    model_trainer = trainer(MLP, epochs, batch_size, lr, train_file, save_dir)
    model_trainer.train()
    
    # Evaluate
    print("\nModel Evaluation on test set:")
    model_trainer.evaluate()

    # Test
    print("\nLoading model and making predictions...")
    model_trainer.test(test_file, os.path.join(output_dir, output_name))

@cli.command()
def ceshi():
    X, y, _, _ = get_input_data("train.csv", is_train=True)
    print(X)
    print(y)


if __name__ == "__main__":
    cli()