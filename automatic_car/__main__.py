import sys
import gymnasium as gym

ENV_NAME = "CarRacing-v2"
ENV_ARGUMENTS = {
    "domain_randomize": False,
    "continuous": False,
    "render_mode": "human",
}


def play(env, model):
    seed = 2000
    obs, _ = env.reset(seed=seed)

    # drop initial frames
    action0 = 0
    for i in range(50):
        obs, _, _, _, _ = env.step(action0)

    done = False
    while not done:
        p = 3  # model.predict(obs)  # adapt to your model
        action = p  # np.argmax(p)  # adapt to your model
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated


def main():
    env = gym.make(ENV_NAME, **ENV_ARGUMENTS)
    print(type(env))
    print("Environment:", env_name)
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)


main()
# # your trained
# model = ...  # your trained model

# play(env, model)
