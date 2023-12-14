import pygame
import gymnasium as gym

# Initialize action
action = 0
quit = False


def play_interactive(env, render_mode="human"):
    def __register_input():
        global quit, action

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action = 1
                if event.key == pygame.K_RIGHT:
                    action = 2
                if event.key == pygame.K_UP:
                    action = 3
                if event.key == pygame.K_DOWN:
                    action = 4  # set 1.0 for wheels to block to zero rotation
                if event.key == pygame.K_ESCAPE:
                    quit = True

            if event.type == pygame.KEYUP:
                action = 0

            if event.type == pygame.QUIT:
                quit = True

    seed = 2000
    env.reset(seed=seed)
    # drop initial frames
    action0 = 0
    for _ in range(50):
        env.step(action0)
    while not quit:
        if render_mode == "human":
            __register_input()
        _, _, terminated, truncated, _ = env.step(action)
    env.close()


if __name__ == "__main__":
    ENV_NAME = "CarRacing-v2"
    ENV_ARGUMENTS = {
        "domain_randomize": False,
        "continuous": False,
        "render_mode": "human",
    }
    env = gym.make(ENV_NAME, **ENV_ARGUMENTS)
    play_interactive(env)
