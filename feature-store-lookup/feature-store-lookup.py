def feature_store_lookup(feature_store: dict, requests: list, defaults: dict) -> list:
    """
    Returns a list of feature dictionaries.
    """
    result = []

    for request in requests:
        user_id = request["user_id"]

        # Copy offline features so inputs are not modified
        offline = feature_store.get(user_id, defaults).copy()

        # Add online features to form the combined vector
        offline.update(request["online_features"])

        result.append(offline)

    return result