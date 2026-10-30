def serialize_user(user):
    return {
        "id": user.user_id,
        "display_name": user.name,
        "email": user.email,
    }
