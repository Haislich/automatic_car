def play_automatic(env, model):
    current_state, _ = env.reset(seed=2000)

    # drop initial frames
    action0 = 0
    for i in range(50):
        current_state, _, _, _, _ = env.step(action0)

    done = False
    while not done:
        p = model.predict(current_state)  # adapt to your model
        action = p  # np.argmax(p)  # adapt to your model
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
