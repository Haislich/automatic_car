import pygame

# Initialize action
action = 0
quit = False
SEED = 2000


def play_automatic(env, model):
    def __register_input():
        global quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit = True

    current_state, _ = env.reset(seed=SEED)

    # drop initial frames

    for _ in range(50):
        current_state, _, _, _, _ = env.step(0)

    done = False
    while not done:
        p = model.predict(current_state)  # adapt to your model
        print(p)
        action = p  # np.argmax(p)  # adapt to your model
        current_state, _, terminated, truncated, _ = env.step(action)
        __register_input()
        done = terminated or truncated or quit
    env.close()


def play_interactive(env):
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

    env.reset(seed=SEED)

    for _ in range(50):
        env.step(0)
    done = False
    while not done:
        __register_input()
        _, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated or quit
    env.close()
