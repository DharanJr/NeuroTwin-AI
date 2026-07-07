def calculate_confidence(predictions):

    total = sum(
        abs(v) for v in predictions.values()
    )

    confidence = 80 + min(total * 2, 19)

    return round(confidence, 1)