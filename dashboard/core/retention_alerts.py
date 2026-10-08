"""
Turns a churn prediction into an actionable, personalized retention
message — instead of just a probability number sitting in a spreadsheet.

Offer selection is simple, transparent rule-based logic (not another ML
model) so it's easy to explain and tune: which offer fits depends on the
customer's contract type, tenure, and monthly spend — the same signals
that tend to drive churn risk for this dataset.
"""

import pandas as pd


def choose_offer(customer: dict) -> dict:
    """Returns {'offer_type': ..., 'offer_text': ...} based on simple,
    explainable rules about the customer's situation."""

    contract = customer.get("Contract", "")
    monthly_charges = float(customer.get("MonthlyCharges", 0))
    tenure = int(customer.get("tenure", 0))
    contract_type_addons = customer.get("OnlineSecurity", "No") == "No" and \
        customer.get("TechSupport", "No") == "No"

    if contract == "Month-to-month":
        return {
            "offer_type": "Contract lock-in discount",
            "offer_text": (
                "10% off for the next 12 months if you switch to an annual "
                "plan — locks in your rate and saves you money long-term."
            ),
        }

    if tenure < 6:
        return {
            "offer_type": "New customer loyalty bonus",
            "offer_text": (
                "A welcome credit worth one month's bill, plus a free trial "
                "of Online Security for 3 months — thank you for being a "
                "recent customer."
            ),
        }

    if monthly_charges > 80:
        return {
            "offer_type": "Bundle discount",
            "offer_text": (
                "15% off your current bundle if you add Online Backup or "
                "Device Protection — often cheaper than paying separately, "
                "and covers you better."
            ),
        }

    if contract_type_addons:
        return {
            "offer_type": "Free add-on trial",
            "offer_text": (
                "A free 3-month trial of Online Security and Tech Support — "
                "customers with these add-ons churn noticeably less."
            ),
        }

    return {
        "offer_type": "General retention check-in",
        "offer_text": (
            "A personal check-in call to see if your current plan still "
            "fits your needs, with a small loyalty credit as a thank-you."
        ),
    }


def build_retention_message(customer: dict, probability: float, risk_level: str) -> dict:
    """Returns {'subject': ..., 'body': ...} — a ready-to-send draft."""
    offer = choose_offer(customer)

    subject = f"A quick thank-you and something special for you"

    tenure = customer.get("tenure")
    tenure_clause = f" for the past {tenure} months" if tenure is not None else ""

    body = (
        f"Hi,\n\n"
        f"We wanted to reach out because we value having you as a customer "
        f"on your {customer.get('Contract', 'current')} plan{tenure_clause}.\n\n"
        f"As a thank-you, we'd like to offer you:\n"
        f"  → {offer['offer_type']}: {offer['offer_text']}\n\n"
        f"If you'd like to take advantage of this, just reply to this email "
        f"or give us a call — we're happy to help.\n\n"
        f"Best,\nCustomer Retention Team\n\n"
        f"---\n"
        f"[Internal note — not sent to customer] Churn risk: {probability:.1%} "
        f"({risk_level}). Offer selected based on: contract type, tenure, "
        f"monthly spend, and current add-ons."
    )

    return {"subject": subject, "body": body, "offer_type": offer["offer_type"]}


def build_alerts_for_batch(
    df_with_predictions: pd.DataFrame,
    probability_col: str = "churn_probability",
    threshold: float = 0.6,
) -> pd.DataFrame:
    """
    Generates one retention message per row flagged at or above `threshold`.
    Returns a DataFrame with subject/body/offer_type columns added, for the
    flagged customers only (sorted by risk, highest first).
    """
    flagged = df_with_predictions[df_with_predictions[probability_col] >= threshold].copy()
    flagged = flagged.sort_values(probability_col, ascending=False)

    subjects, bodies, offer_types = [], [], []
    for _, row in flagged.iterrows():
        customer = row.to_dict()
        probability = customer[probability_col]
        risk_level = "High" if probability >= 0.6 else "Medium"
        message = build_retention_message(customer, probability, risk_level)
        subjects.append(message["subject"])
        bodies.append(message["body"])
        offer_types.append(message["offer_type"])

    flagged["offer_type"] = offer_types
    flagged["message_subject"] = subjects
    flagged["message_body"] = bodies

    return flagged
