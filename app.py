
import re
from datetime import datetime, timedelta, date

import pandas as pd
import gradio as gr

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity


# =====================================
# LOAD AND PREPARE LUNA FAQ DATASET
# =====================================

df = pd.read_csv("Luna's Chatbot.csv")
df = df.dropna(how="all")

for col in ["QUESTION", "CATEGORY", "ANSWER"]:
    df[col] = df[col].fillna("").astype(str).str.strip()

df = df[
    (df["QUESTION"] != "") &
    (df["CATEGORY"] != "") &
    (df["ANSWER"] != "")
].reset_index(drop=True)

category_mapping = {
    "Cycle": "Menstrual Cycle",
    "Cramps": "Cramps and Pain",
    "PMS": "PMS and Hormonal Changes",
    "Hygiene": "Menstrual Hygiene",
    "Self-Care": "Self-care and Lifestyle",
    "Other": "Other / Fallback"
}

df["CATEGORY"] = df["CATEGORY"].replace(category_mapping)

# Train the question classification model
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(df["QUESTION"], df["CATEGORY"])

# Prepare FAQ similarity search
faq_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
faq_vectors = faq_vectorizer.fit_transform(df["QUESTION"])


# =====================================
# FAQ QUESTION ANSWERING
# =====================================

def ask_luna(user_question):

    predicted_category = model.predict([user_question])[0]

    candidates = df[df["CATEGORY"] == predicted_category]

    user_vector = faq_vectorizer.transform([user_question])

    candidate_vectors = faq_vectorizer.transform(
        candidates["QUESTION"]
    )

    similarities = cosine_similarity(
        user_vector, candidate_vectors
    )[0]

    best_index = similarities.argmax()
    best_score = similarities[best_index]

    if best_score < 0.12:
        return (
            "I'm not sure I understood that question. "
            "Please ask me something about menstrual health or periods."
        )

    return candidates.iloc[best_index]["ANSWER"]


# =====================================
# MENSTRUAL CYCLE CALCULATOR
# =====================================

def calculate_period(last_period_date, cycle_length):

    try:
        start_date = datetime.strptime(
            last_period_date.strip(), "%Y-%m-%d"
        ).date()

        if start_date > date.today():
            return "Please enter a date that is not in the future."

        if cycle_length is None:
            return "Please enter your average cycle length."

        cycle_length = int(cycle_length)

        if not 1 <= cycle_length <= 90:
            return "Please enter a cycle length between 1 and 90 days."

        next_period = start_date + timedelta(days=cycle_length)

        ovulation = next_period - timedelta(days=14)

        fertile_start = ovulation - timedelta(days=5)
        fertile_end = ovulation

        return f"""
🌙 LUNA'S CYCLE PREDICTIONS

🩸 NEXT EXPECTED PERIOD
{next_period.strftime('%A, %d %B %Y')}

🥚 ESTIMATED OVULATION
{ovulation.strftime('%A, %d %B %Y')}

🌱 ESTIMATED FERTILE WINDOW
{fertile_start.strftime('%d %B %Y')} to
{fertile_end.strftime('%d %B %Y')}

📅 AVERAGE CYCLE LENGTH
{cycle_length} days

IMPORTANT:
These are calendar-based estimates.
Actual period and ovulation dates may vary.
Do not use these predictions as contraception.
"""

    except ValueError:
        return "Please enter the date in YYYY-MM-DD format, e.g. 2026-10-10."


# =====================================
# CHATBOT CONNECTION
# =====================================

def chat_with_luna(message, history):

    if not message.strip():
        return "Please type a question first."

    message_lower = message.lower()

    tracking_keywords = [
        "track my period",
        "calculate my period",
        "predict my period",
        "next period",
        "ovulation",
        "fertile window",
        "cycle prediction"
    ]

    if any(word in message_lower for word in tracking_keywords):

        date_match = re.search(
            r"\b\d{4}-\d{2}-\d{2}\b", message
        )

        cycle_match = re.search(
            r"(\d+)\s*(?:day|days)",
            message_lower
        )

        if not date_match or not cycle_match:
            return (
                "🌙 I can calculate your cycle predictions!\n\n"
                "Please provide:\n"
                "1. First day of your last period (YYYY-MM-DD)\n"
                "2. Your average cycle length in days\n\n"
                "Example: My last period started on "
                "2026-10-10 and my cycle is 28 days. Track my period."
            )

        return calculate_period(
            date_match.group(),
            int(cycle_match.group(1))
        )

    return ask_luna(message)


# =====================================
# LUNA GRADIO INTERFACE
# =====================================

with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🌙 Luna — Your Period Health Companion

    Ask questions about menstrual health, periods, cramps,
    PMS, hygiene and cycle tracking.
    """)

    with gr.Tab("Chat with Luna"):

        gr.ChatInterface(
            fn=chat_with_luna,
            examples=[
                "Why do periods happen?",
                "Why do I get cramps?",
                "What is PMS?",
                "How often should I change my pad?",
                "My last period started on 2026-10-10 and my cycle is 28 days. Track my period."
            ],
            title="Chat with Luna"
        )

    with gr.Tab("Period Calculator"):

        gr.Markdown("### 🌸 Calculate Your Menstrual Cycle")

        date_input = gr.Textbox(
            label="First day of your last period",
            placeholder="YYYY-MM-DD"
        )

        cycle_input = gr.Number(
            label="Average cycle length (days)",
            value=28,
            precision=0
        )

        calculate_button = gr.Button(
            "Calculate My Cycle",
            variant="primary"
        )

        result = gr.Textbox(
            label="Your Cycle Predictions",
            lines=14
        )

        calculate_button.click(
            fn=calculate_period,
            inputs=[date_input, cycle_input],
            outputs=result
        )


demo.launch(server_name="0.0.0.0", server_port=7860)
