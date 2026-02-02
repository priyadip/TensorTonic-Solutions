def retraining_policy(daily_stats, config):
    """
    Determines which days to trigger retraining based on drift, performance, and staleness.
    
    Args:
        daily_stats (list[dict]): A list of dictionaries containing 'day', 'drift_score', and 'performance'.
        config (dict): A dictionary containing thresholds and constraints.
        
    Returns:
        list[int]: A sorted list of day numbers where retraining occurred.
    """
    
    # Extract configuration
    drift_threshold = config["drift_threshold"]
    performance_threshold = config["performance_threshold"]
    max_staleness = config["max_staleness"]
    cooldown = config["cooldown"]
    retrain_cost = config["retrain_cost"]
    budget = config["budget"]

    retrain_days = []

    # --- State Initialization ---
    
    # 1. Staleness State: Model age
    # Constraint: "days_since_retrain starts at 0"
    days_since_retrain = 0

    # 2. Cooldown State: Day of last retrain
    # Constraint: "Cooldown is initially satisfied"
    # To satisfy (Day 1 - last_retrain_day) >= cooldown, 
    # we initialize last_retrain_day to a value far in the past.
    last_retrain_day = 1 - cooldown

    for stat in daily_stats:
        current_day = stat["day"]
        drift_score = stat["drift_score"]
        performance = stat["performance"]

        # Increment staleness at the start of every day
        days_since_retrain += 1

        # --- Trigger Check (OR Logic) ---
        is_drift = drift_score > drift_threshold
        is_perf_drop = performance < performance_threshold
        is_stale = days_since_retrain >= max_staleness
        
        trigger_signal = is_drift or is_perf_drop or is_stale

        # --- Constraint Check (AND Logic) ---
        passed_cooldown = (current_day - last_retrain_day) >= cooldown
        has_budget = budget >= retrain_cost

        # --- Decision ---
        if trigger_signal and passed_cooldown and has_budget:
            # Execute Retrain
            retrain_days.append(current_day)
            
            # Update System State
            budget -= retrain_cost
            days_since_retrain = 0         # Reset model age
            last_retrain_day = current_day # Reset cooldown timer

    return retrain_days