def normalize_users(rows):
    users = []
    for row in rows:
        users.append({
            "email": row["email"],
            "name": row.get("name", ""),
            "active": row.get("active", True),
        })
    return users
