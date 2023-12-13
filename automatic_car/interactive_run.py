import gymnasium as gym
from pynput import keyboard

# Initialize action and environment
# ENV_NAME = "CarRacing-v2"
# ENV_ARGUMENTS = {
#     "domain_randomize": False,
#     "continuous": False,
#     "render_mode": "human",
# }
action = [0.0, 0.0, 0.0, 0.0, 0.0]
# env = gym.make(ENV_NAME, **ENV_ARGUMENTS)
# observation = env.reset()


# Define on_press and on_release functions
def on_press(key):
    print("on press", key)
    try:
        if key.char == "a":  # turn left
            action[0] = -1.0
        elif key.char == "d":  # turn right
            action[0] = 1.0
        elif key.char == "w":  # accelerate
            action[1] = 1.0
        elif key.char == "s":  # brake
            action[2] = 1.0
    except AttributeError:
        pass


def on_release(key):
    print(key)
    try:
        if key.char == "a" and action[0] == -1.0:
            action[0] = 0.0
        elif key.char == "d" and action[0] == 1.0:
            action[0] = 0.0
        elif key.char == "w":
            action[1] = 0.0
        elif key.char == "s":
            action[2] = 0.0
    except AttributeError:
        pass


# Collect events until released
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# Run the game loop
while True:
    pass
    # env.render()
    # obs, _, terminated, truncated, _ = env.step(action)
    # done = terminated or truncated
    # if done:
    #     observation = env.reset()
