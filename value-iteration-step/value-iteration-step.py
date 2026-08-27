def value_iteration_step(values: list, transitions: list, rewards: list, gamma: float) -> list[float]:
    """
    Returns one updated floating-point value for every state.
    """
    updated_values = []

    for s in range(len(values)):
        action_values = []

        for a in range(len(rewards[s])):
            expected_value = sum(
                transitions[s][a][next_s] * values[next_s]
                for next_s in range(len(values))
            )

            q_value = rewards[s][a] + gamma * expected_value
            action_values.append(q_value)

        updated_values.append(float(max(action_values)))

    return updated_values