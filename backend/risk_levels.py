def get_risk_label(score, inverse=False):
    if inverse:
        score = -score

    if score <= -1:
        return "Low"
    elif score <= 1:
        return "Moderate"
    else:
        return "High"