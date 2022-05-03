import argparse
import os
from types import SimpleNamespace
import wandb
import yaml
import DRCSR


def get_sweep(epochs, path, model_name, wandb_api_key):
    return lambda: sweep(epochs, path, model_name, wandb_api_key)


def sweep(epochs, path, model_name, wandb_api_key):
    with wandb.init():
        config = wandb.config

        simple_namespace = SimpleNamespace(
            epochs=epochs,
            path=path,
            model_name=model_name,
            wandb_api_key=wandb_api_key
        )

        drcsr_model = DRCSR.DRCSR_model(simple_namespace, config)
        drcsr_model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", "-e", type=int, required=True)
    parser.add_argument("--path", "-p", type=str, required=True)
    parser.add_argument("--model_name", "-m", type=str, required=True)
    parser.add_argument("--project_name", "-pn", type=str, required=True)
    parser.add_argument("--runs", "-r", type=int, required=True)
    parser.add_argument("--wandb_api_key", "-wak", type=str, required=True)
    parser.add_argument("--sweep_id", "-wi", type=str, required=True)
    args = parser.parse_args()

    os.environ["WANDB_API_KEY"] = args.wandb_api_key

    sweep_function = get_sweep(
        epochs=args.epochs,
        path=args.path,
        model_name=args.model_name,
        wandb_api_key=args.wandb_api_key
    )

    print("START SWEEPING THE FLOOR NOW!")
    wandb.agent(args.sweep_id, function=sweep_function, count=args.runs, project=args.project_name)
