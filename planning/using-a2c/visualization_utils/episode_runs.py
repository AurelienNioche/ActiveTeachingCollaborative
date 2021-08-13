def run_one_episode(env, policy, user=None, zero_penalty_coeff=False, is_a2c=True):
    rewards = []
    actions = []
    if user is not None:
        obs = env.reset(user)
    else:
        obs = env.reset(env.current_user)
    pc = env.penalty_coeff
    if zero_penalty_coeff:
        env.penalty_coeff=0.0
    while True:
        if is_a2c:
            action = policy.predict(obs, deterministic=True)
        else:
            action = policy.act(obs)
        obs, reward, done, info = env.step(action)
        rewards.append(reward / env.reward_coeff)
        actions.append(action)
        if done:
            obs = env.reset(env.current_user)
            break
    env.penalty_coeff = pc
    return rewards, actions
