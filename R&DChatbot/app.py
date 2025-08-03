from flask import Flask, request, jsonify, render_template
from multi_rag import MultiRAG
from pymongo import MongoClient
import requests
import re

app = Flask(__name__)

# RAG from PDFs
rag = MultiRAG(pdf_folder="docs", db_path="faiss_index_pdf")

# MongoDB for course listings
mongo_client = MongoClient("mongodb+srv://rnd:rnduser@cluster0.5lkna8o.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
courses_collection = mongo_client["BCIS_Courses"]["courses"]

OLLAMA_API_URL = "http://127.0.0.1:11434/api/chat"
MODEL_NAME = "deepseek-r1:8b"

def clean_ollama_response(text):
    return re.sub(r"<[^>]+>", "", text).strip()

def extract_last_paragraph(text):
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    return paragraphs[-1] if paragraphs else text.strip()

KNOWN_MAJORS = [
    "Software Development", "Computer Science",
    "Data Science", "Networks and Cybersecurity", "Digital Services"
]

def detect_major(text):
    for major in KNOWN_MAJORS:
        if major.lower() in text.lower():
            return major
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    print("💬 User:", user_input)

    if re.search(r"study\s*plan|course\s*list", user_input.lower()):
        detected_major = detect_major(user_input) or "Software Development"
        match_year = re.search(r"year\s*([1-3])", user_input.lower())
        filter_year = int(match_year.group(1)) if match_year else None

        query = { "majors": { "$in": [detected_major] } }
        if filter_year:
            query["year"] = filter_year

        results = list(courses_collection.find(query))
        if not results and filter_year:
            query.pop("year")
            results = list(courses_collection.find(query))

        plan = {}
        for course in results:
            year = course.get('year')
            code = course.get('code')
            title = course.get('title')
            sems = course.get('semester', [])
            label = f"{code}: {title}"
            for sem in sems:
                year_key = f"Year {year}"
                sem_key = f"Semester {sem}"
                plan.setdefault(year_key, {}).setdefault(sem_key, set()).add(label)

        output = f"📘 Study Plan for {detected_major}"
        if filter_year:
            output += f" (Year {filter_year})"
        output += ":\n\n"

        for year in sorted(plan.keys()):
            output += f"{year}:\n"
            for sem in sorted(plan[year].keys()):
                courses = sorted(plan[year][sem])
                output += f"  {sem}: " + ", ".join(courses) + "\n"
            output += "\n"

        return jsonify({ "response": output.strip() })

    # Else: PDF descriptor-based Q&A
    context = rag.retrieve_relevant_context(user_input)
    prompt = f"""You are a helpful assistant. Use the provided context to answer concisely and clearly. Do not guess. Only use what's found in the context.

Context:
{context}

User Question: {user_input}
Answer:"""

    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            }
        )

        full_response = response.json()
        raw_answer = full_response.get('message', {}).get('content', 'No content in response')
        cleaned = clean_ollama_response(raw_answer)
        concise_answer = extract_last_paragraph(cleaned)

        return jsonify({ "response": concise_answer })

    except Exception as e:
        print("Error:", str(e))
        return jsonify({ "response": "An error occurred connecting to Ollama." })

if __name__ == '__main__':
    app.run(debug=True)

