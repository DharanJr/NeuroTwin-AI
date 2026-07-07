from flask import Flask, request, jsonify
from flask_cors import CORS

from predictor import predict_all
from risk_levels import get_risk_label

app = Flask(__name__)
CORS(app)


def validate_request_json():
    """
    Basic validation: ensure request has JSON body and it's a dict.
    Returns (data, error_response). If error_response is not None, caller should return it.
    """
    if not request.is_json:
        return None, (jsonify({
            "success": False,
            "error": "Request must have Content-Type: application/json"
        }), 400)

    data = request.get_json(silent=True)

    if data is None or not isinstance(data, dict):
        return None, (jsonify({
            "success": False,
            "error": "Request body must be a valid JSON object"
        }), 400)

    return data, None


@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "project": "NeuroTwin AI"
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "backend": "working"
    })


@app.route("/predict", methods=["POST"])
def predict():
    user_data, error = validate_request_json()
    if error:
        return error

    try:
        predictions = predict_all(user_data)

        risk_levels = {}

        for key, value in predictions.items():
            if key == "cognitive_age":
                continue
            risk_levels[key] = get_risk_label(
                value,
                inverse=(key == "memory_retention")
            )

        avg_risk = (
            predictions["brain_fog"] +
            predictions["digital_addiction"] +
            predictions["attention_fragmentation"] +
            predictions["digital_overstimulation"]
        ) / 4

        overall_risk = get_risk_label(avg_risk)

        recommendations = []

        if predictions["brain_fog"] > 1:
            recommendations.append(
                "Increase sleep duration and reduce stress."
            )

        if predictions["digital_addiction"] > 1:
            recommendations.append(
                "Reduce screen time and social media usage."
            )

        if predictions["attention_fragmentation"] > 1:
            recommendations.append(
                "Use focused study sessions and fewer notifications."
            )

        if predictions["digital_overstimulation"] > 1:
            recommendations.append(
                "Take regular digital detox breaks."
            )

        if predictions["memory_retention"] < -1:
            recommendations.append(
                "Memory retention is below average — consider spaced repetition and consistent sleep schedules."
            )

        if len(recommendations) == 0:
            recommendations.append(
                "Your cognitive health looks healthy. Maintain current habits."
            )

        return jsonify({
            "success": True,
            "cognitive_age": round(predictions["cognitive_age"], 1),
            "overall_risk": overall_risk,
            "confidence": 94,
            "predictions": predictions,
            "risk_levels": risk_levels,
            "recommendations": recommendations
        })

    except KeyError as e:
        return jsonify({
            "success": False,
            "error": f"Missing or invalid field: {str(e)}"
        }), 400

    except (ValueError, TypeError) as e:
        return jsonify({
            "success": False,
            "error": f"Invalid input value: {str(e)}"
        }), 400

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/simulate", methods=["POST"])
def simulate():
    data, error = validate_request_json()
    if error:
        return error

    base_input = data.get("base_input")
    interventions = data.get("interventions")

    if not isinstance(base_input, dict):
        return jsonify({
            "success": False,
            "error": "'base_input' is required and must be a JSON object"
        }), 400

    if not isinstance(interventions, dict):
        return jsonify({
            "success": False,
            "error": "'interventions' is required and must be a JSON object"
        }), 400

    try:
        before = predict_all(base_input)

        modified = {
            **base_input,
            **interventions
        }

        after = predict_all(modified)

        improvements = {}

        for metric in [
            "brain_fog",
            "digital_addiction",
            "attention_fragmentation",
            "digital_overstimulation",
            "memory_retention"
        ]:
            try:
                before_val = float(before[metric])
                after_val = float(after[metric])

                improvements[metric] = round(
                    after_val - before_val,
                    3
                )

            except Exception:
                improvements[metric] = 0

        return jsonify({
            "success": True,
            "before": before,
            "after": after,
            "improvements": improvements
        })

    except KeyError as e:
        return jsonify({
            "success": False,
            "error": f"Missing or invalid field: {str(e)}"
        }), 400

    except (ValueError, TypeError) as e:
        return jsonify({
            "success": False,
            "error": f"Invalid input value: {str(e)}"
        }), 400

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )