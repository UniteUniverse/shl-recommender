import json
import pandas as pd
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = SentenceTransformer("all-mpnet-base-v2")

def load_assessments():
    df = pd.read_csv("shl_assessments.csv")
    df["test_type"] = df["test_type"].apply(eval)
    df["embedding"] = df.apply(lambda row: model.encode(f"{row['name']} {row['description']}"), axis=1)
    return df.to_dict("records")

assessments = load_assessments()

def parse_query(query: str):
    prompt = f"""
    Extract the following from the query in JSON format (NO MARKDOWN):
    {{
        "skills": ["list", "of", "technical", "skills"],
        "max_duration": integer (strictly from phrases like "X minutes" or "X-hour"),
        "test_types": ["cognitive", "personality", "technical", "language", "behavioral"]
    }}

    **Mapping Guide**:
    - "Teamwork" → test_types: ["behavioral"]
    - "Java/Python" → skills: ["Java", "Python"]
    - "Cultural fit" → test_types: ["personality"]
    - "QA Engineer" → test_types: ["technical"]

    Query: {query}

    """
    try:
        response = genai.GenerativeModel("gemini-2.5-flash-preview-04-17").generate_content(prompt)
        clean_response = response.text.replace("```json", "").replace("```", "").strip()
        print(f"Raw Gemini response: {clean_response}")  # Debug line
        criteria = json.loads(clean_response)
        criteria["max_duration"] = int(criteria["max_duration"]) if criteria.get("max_duration") else None
        return criteria
    except Exception as e:
        print(f"Error parsing query: {e}")  # Debug line
        return {"skills": [], "max_duration": None, "test_types": []}

def recommend_assessments(query: str):
    query_embedding = model.encode(query)
    filtered = []
    for test in assessments:
        similarity = cosine_similarity([query_embedding], [test["embedding"]])[0][0]
        filtered.append({**test, "similarity": similarity})
    filtered.sort(key=lambda x: x["similarity"], reverse=True)
    return filtered[:10]
    