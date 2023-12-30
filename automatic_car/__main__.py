import argparse
import gymnasium as gym
from automatic_car.play import play_interactive
from automatic_car.play import play_automatic
from automatic_car.models import Model1
from automatic_car.models import Model2
import torch
import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
ENV_NAME = "CarRacing-v2"
ENV_ARGUMENTS = {
    "domain_randomize": False,
    "continuous": False,
    "render_mode": "human",
}


def main():
    env = gym.make(ENV_NAME, **ENV_ARGUMENTS)
    parser = argparse.ArgumentParser(
        prog="Automatic car",
        description="This cli launches an instance of CarRacing-v2, that plays using a CNN.",
    )
    parser.add_argument(
        "model",
        metavar="model",
        type=str,
        nargs="?",
        choices=["model1", "model2"],
        help="an integer for the accumulator",
    )
    parser.add_argument(
        action="store_const",
        default=True,
        dest="interactive",
        help="Interactively drive the car using the arrow keys.",
    )

    args = parser.parse_args()
    if args.model:
        if args.model == "model1":
            model = Model1(3, 5)
            torch.load(model.path)
            play_automatic(env=env, model=model)
        elif args.model == "model2":
            model = Model2(3, 5)
            torch.load(model.path)
            play_automatic(env=env, model=model)
    # If no model is specified the app is launched in interactive mode.
    else:
        play_interactive(env=env)


if __name__ == "__main__":
    main()
