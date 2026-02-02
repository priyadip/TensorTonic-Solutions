def value_iteration_step(values, transitions, rewards, gamma):
    new_values = []

    num_states = len(values)

    for s in range(num_states):
        best = float("-inf")

        for a in range(len(rewards[s])):
            q = rewards[s][a]

            future = 0.0
            for s_next in range(num_states):
                future += transitions[s][a][s_next] * values[s_next]

            q += gamma * future
            best = max(best, q)

        new_values.append(best)

    return new_values
