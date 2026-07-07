def generate_recommendations(user_data, predictions):

    recommendations = []

    if user_data["sleep_hours"] < 7:
        recommendations.append(
            "Increase sleep duration to 7-8 hours daily."
        )

    if user_data["daily_screen_time_hours"] > 6:
        recommendations.append(
            "Reduce screen time to below 6 hours."
        )

    if user_data["exercise_minutes"] < 30:
        recommendations.append(
            "Exercise at least 30 minutes per day."
        )

    if user_data["social_media_hours"] > 3:
        recommendations.append(
            "Reduce social media usage."
        )

    if predictions["brain_fog"] > 1:
        recommendations.append(
            "Brain fog risk is elevated. Improve sleep and reduce multitasking."
        )

    if predictions["digital_addiction"] > 1:
        recommendations.append(
            "Digital addiction risk is high. Consider digital detox periods."
        )

    if predictions["attention_fragmentation"] > 1:
        recommendations.append(
            "Frequent distractions detected. Reduce phone interruptions."
        )

    if predictions["memory_retention"] < 0:
        recommendations.append(
            "Memory retention is low. Focus on active recall and quality sleep."
        )

    return recommendations