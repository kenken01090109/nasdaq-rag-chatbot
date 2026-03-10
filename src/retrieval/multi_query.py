from google import genai

client = genai.Client()

def generate_queries(query: str) -> list[str]:
    prompt = f"""
Generate 3 alternative search queries for the following user question.
Return only the 3 queries, one per line, with no numbering.

User question:
{query}
"""

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )

    text = response.text if response.text else ""
    queries = [q.strip() for q in text.split("\n") if q.strip()]
    return queries[:3]