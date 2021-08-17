import numpy as np
from tqdm import tqdm


def run_one_episode(env, policy, user=None):
    rewards = []
    actions = []

    obs = env.reset(user)

    with tqdm(total=env.n_iter_per_session * env.n_session) as pb:
        while True:
            action = policy.act(obs)
            obs, reward, done, _ = env.step(action)
            rewards.append(reward)
            actions.append(action)
            if done:
                # Simulate exam
                obs, reward, done, _ = env.step(None)
                rewards.append(reward)
                break

            pb.update()

    final_n_learned = reward * env.n_item
    n_view = len(np.unique(np.asarray(actions)))
    print(f"{policy.__class__.__name__.lower()} | "
          f"final reward {int(final_n_learned)} | "
          f"precision {final_n_learned / n_view:.2f}")
    return rewards, actions
