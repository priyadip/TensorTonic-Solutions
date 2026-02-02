def schedule_pipeline(tasks, resource_budget):
    # Normalize task info
    task_map = {}
    for t in tasks:
        task_map[t["name"]] = {
            "duration": t["duration"],
            "resources": t["resources"],
            # FIX: Use "depends_on" instead of "deps" to match input format
            "deps": set(t.get("depends_on", [])) 
        }

    completed = set()
    started = set()
    running = {}        # task_name -> end_time
    start_times = {}

    time = 0
    used_resources = 0

    # Continue until all tasks are completed
    while len(completed) < len(task_map):

        # 1. Complete finished tasks
        # Find tasks that finish exactly at current 'time'
        finished = [t for t, end in running.items() if end == time]
        for t in finished:
            used_resources -= task_map[t]["resources"]
            completed.add(t)
            del running[t]

        # 2. Identify ready tasks
        ready = []
        for name, info in task_map.items():
            # Must not have started yet
            if name in started:
                continue
            # Dependencies must be fully completed
            if info["deps"].issubset(completed):
                ready.append(name)

        # 3. Sort alphabetically
        ready.sort()

        # 4. Greedily assign tasks
        for name in ready:
            req = task_map[name]["resources"]
            # Check budget constraint
            if used_resources + req <= resource_budget:
                started.add(name)
                start_times[name] = time
                used_resources += req
                running[name] = time + task_map[name]["duration"]

        # 5. Advance time to next completion event
        # If tasks are running, jump to the earliest finish time.
        # If nothing is running and we aren't done, it's a deadlock (shouldn't happen per constraints).
        if running:
            time = min(running.values())
        elif len(completed) < len(task_map):
            # Fallback if nothing runs but work remains (impossible in valid DAG)
            break 

    # Output sorted by (start_time, task_name)
    result = [(name, start_times[name]) for name in start_times]
    result.sort(key=lambda x: (x[1], x[0]))
    return result