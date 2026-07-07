def calculate_cognitive_age(
    actual_age,
    brain_fog,
    addiction,
    attention,
    overstimulation,
    memory
):
    risk_score = (
        (brain_fog * 1.5)
        + (addiction * 1.2)
        + (attention * 1.2)
        + (overstimulation * 1.0)
        - (memory * 1.5)
    )

    cognitive_age = actual_age + risk_score

    if cognitive_age < actual_age - 5:
        cognitive_age = actual_age - 5

    if cognitive_age > actual_age + 20:
        cognitive_age = actual_age + 20

    return round(cognitive_age, 1)