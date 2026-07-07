def calculate_overall_risk(predictions):

    score = (
        predictions["brain_fog"]
        + predictions["digital_addiction"]
        + predictions["attention_fragmentation"]
        + predictions["digital_overstimulation"]
        - predictions["memory_retention"]
    )

    if score < 2:
        return "Low"

    elif score < 6:
        return "Moderate"

    return "High"