from __future__ import annotations

import os
import re
from typing import Any

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from neo4j import GraphDatabase


app = Flask(__name__)
VECTOR_MIN_SCORE = 0.60
UNSUPPORTED_TOPIC_TERMS = {
    "visa",
    "visas",
    "immigration",
    "sponsor",
    "sponsorship",
    "refund",
    "deadline",
    "deadlines",
    "accommodation",
    "guarantee",
    "guaranteed",
}
UNKNOWN_ANSWER = "I don't know from the current course data."


DIRECT_QUERIES = {
    "summary": """
        MATCH (course:Course)
        RETURN course.title AS title,
               course.institution AS institution,
               course.course_code AS code,
               course.campus AS campus,
               course.duration AS duration,
               course.delivery AS delivery,
               course.courseStatus AS status,
               course.sourceUrl AS sourceUrl
    """,
    "modules": """
        MATCH (:Course)-[rel]->(module:Module)
        WHERE type(rel) IN ["HAS_CORE_MODULE", "HAS_OPTIONAL_MODULE"]
        RETURN CASE type(rel)
          WHEN "HAS_CORE_MODULE" THEN "Core"
          ELSE "Optional"
        END AS type,
        module.name AS name
        ORDER BY type, name
    """,
    "careers": """
        MATCH (:Course)-[:LEADS_TO_ROLE]->(role:CareerRole)
        RETURN role.name AS role
        ORDER BY role
    """,
    "fees": """
        MATCH (:Course)-[:HAS_SECTION]->(:Section {sectionKey: "fees"})-[:HAS_CHUNK]->(chunk:Chunk)
        RETURN chunk.text AS text
        ORDER BY chunk.chunkSeqId
    """,
    "entry": """
        MATCH (:Course)-[:HAS_SECTION]->(:Section {sectionKey: "entry"})-[:HAS_CHUNK]->(chunk:Chunk)
        RETURN chunk.text AS text
        ORDER BY chunk.chunkSeqId
    """,
    "teaching": """
        MATCH (:Course)-[:HAS_SECTION]->(:Section {sectionKey: "learning_and_teaching"})-[:HAS_CHUNK]->(chunk:Chunk)
        RETURN chunk.text AS text
        ORDER BY chunk.chunkSeqId
    """,
    "contact": """
        MATCH (:Course)-[:CONTACT]->(contact:Contact)
        RETURN contact.institution AS institution,
               contact.campus AS campus,
               contact.address AS address,
               contact.switchboard AS switchboard
    """,
}


def env_ready() -> dict[str, bool]:
    load_dotenv(".env", override=True)
    openai_key = os.getenv("OPENAI_API_KEY", "")
    groq_key = os.getenv("GROQ_API_KEY", "")
    return {
        "neo4j": all(
            os.getenv(key)
            for key in ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "NEO4J_DATABASE"]
        ),
        "openai": bool(openai_key and not openai_key.startswith("replace_with")),
        "groq": bool(groq_key and not groq_key.startswith("replace_with")),
        "vector": vector_ready(),
    }


def run_cypher(query: str, **params) -> list[dict[str, Any]]:
    load_dotenv(".env", override=True)
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    try:
        with driver.session(database=os.getenv("NEO4J_DATABASE") or "neo4j") as session:
            return session.run(query, **params).data()
    finally:
        driver.close()


def run_query(query_name: str) -> list[dict[str, Any]]:
    return run_cypher(DIRECT_QUERIES[query_name])


def vector_ready() -> bool:
    try:
        rows = run_cypher(
            """
            MATCH (chunk:Chunk)
            WHERE chunk.courseId = "uwe-msc-data-science-inb112"
              AND chunk.textEmbedding IS NOT NULL
            RETURN count(chunk) AS embeddedChunks
            """
        )
    except Exception:
        return False
    return bool(rows and rows[0]["embeddedChunks"] > 0)


def classify_question(question: str) -> str:
    normalized = question.lower()
    if any(word in normalized for word in ["fee", "cost", "price", "tuition", "money"]):
        return "fees"
    if any(word in normalized for word in ["entry", "requirement", "ielts", "apply", "application"]):
        return "entry"
    if any(word in normalized for word in ["module", "course structure", "optional", "core"]):
        return "modules"
    if any(word in normalized for word in ["career", "job", "role", "graduate", "work"]):
        return "careers"
    if any(word in normalized for word in ["teach", "learning", "assessment", "study"]):
        return "teaching"
    if any(word in normalized for word in ["contact", "phone", "address", "campus", "where"]):
        return "contact"
    return "summary"


def format_direct_answer(query_name: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "I could not find that information in the current graph."

    if query_name == "summary":
        row = rows[0]
        return (
            f"{row['title']} is a {row['institution']} postgraduate course with code "
            f"{row['code']}. It is based at {row['campus']} and is currently "
            f"{row['status']}. Duration: {row['duration']} Delivery: {row['delivery']}"
        )

    if query_name == "modules":
        core = [row["name"] for row in rows if row["type"] == "Core"]
        optional = [row["name"] for row in rows if row["type"] == "Optional"]
        return (
            "Core modules: "
            + "; ".join(core)
            + "\n\nOptional modules: "
            + "; ".join(optional)
        )

    if query_name == "careers":
        return "Listed career roles: " + "; ".join(row["role"] for row in rows)

    if query_name == "contact":
        row = rows[0]
        return (
            f"{row['institution']}, {row['campus']}. Address: {row['address']}. "
            f"Switchboard: {row['switchboard']}."
        )

    return "\n\n".join(row["text"] for row in rows)


def answer_with_llm(question: str) -> str:
    from chat_uwe_kg import answer as llm_answer

    return llm_answer(question, mode="vector")


def format_model_answer(text: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</?[^>]+>", "", text)
    text = text.replace("**", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def course_answer_style_prompt(context_type: str) -> str:
    return (
        "You answer questions about the UWE Bristol MSc Data Science course. "
        f"Use only the {context_type}. If the context does not contain the answer, "
        "say you do not know from the current course data. "
        "Do not infer unsupported policies, deadlines, visa information, scholarships, "
        "admissions decisions, eligibility decisions, refunds, accommodation details, "
        "or guarantees. "
        "Format for a plain-text chat UI: do not use Markdown tables, HTML tags, "
        "or raw markup. Use a short opening sentence, then simple bullet points "
        "with '- ' when helpful. Keep section headings short and plain."
    )


def refine_user_question(question: str) -> str:
    from groq import Groq

    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        max_completion_tokens=180,
        top_p=1,
        reasoning_effort="medium",
        stream=False,
        stop=None,
        messages=[
            {
                "role": "system",
                "content": (
                    "Rewrite the user's message into a clear, grammatically correct "
                    "English question for searching UWE Bristol MSc Data Science course "
                    "information. Preserve the original meaning. Do not answer the "
                    "question. Do not add facts, assumptions, constraints, or new topics. "
                    "Return plain text only, with one rewritten question."
                ),
            },
            {"role": "user", "content": question},
        ],
    )
    refined = format_model_answer(completion.choices[0].message.content)
    return refined or question


def answer_with_groq(question: str) -> str:
    from groq import Groq

    query_name = classify_question(question)
    context = format_direct_answer(query_name, run_query(query_name))
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    completion = client.chat.completions.create(
        model=model,
        temperature=0.1,
        max_completion_tokens=2048,
        top_p=1,
        reasoning_effort="medium",
        stream=False,
        stop=None,
        messages=[
            {
                "role": "system",
                "content": course_answer_style_prompt("provided graph context"),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nGraph context:\n{context}",
            },
        ],
    )
    return format_model_answer(completion.choices[0].message.content)


def extract_unsupported_terms(question: str) -> set[str]:
    normalized = question.lower()
    return {term for term in UNSUPPORTED_TOPIC_TERMS if term in normalized}


def context_mentions_terms(context: str, terms: set[str]) -> bool:
    normalized = context.lower()
    return any(term in normalized for term in terms)


def unique_sources(rows: list[dict[str, Any]]) -> list[str]:
    sources: list[str] = []
    for row in rows:
        section = row.get("section")
        if section and section not in sources:
            sources.append(section)
    return sources


def answer_with_vector_groq(question: str) -> tuple[str, list[str]]:
    context_rows = run_cypher(
        """
        CALL db.index.vector.queryNodes($indexName, $topK, $questionEmbedding)
          YIELD node, score
        WHERE score >= $minScore
        OPTIONAL MATCH window=(:Chunk)-[:NEXT*0..1]->(node)-[:NEXT*0..1]->(:Chunk)
        WITH node, score, window
          ORDER BY score DESC, length(window) DESC
        WITH node, score, head(collect(window)) AS longestWindow
        WITH node, score,
          CASE
            WHEN longestWindow IS NULL THEN [node]
            ELSE nodes(longestWindow)
          END AS chunkList
        UNWIND chunkList AS chunk
        WITH node, score, chunk
          ORDER BY score DESC, chunk.sectionId ASC, chunk.chunkSeqId ASC
        WITH node, score, collect(chunk.text) AS textList
        RETURN node.sectionTitle AS section,
               score,
               textList
        ORDER BY score DESC
        """,
        indexName="uwe_course_chunks",
        topK=4,
        minScore=VECTOR_MIN_SCORE,
        questionEmbedding=embed_text(question),
    )
    if not context_rows:
        return UNKNOWN_ANSWER, []

    context = "\n\n".join(
        f"Section: {row['section']}\n" + "\n\n".join(row["textList"])
        for row in context_rows
    )
    unsupported_terms = extract_unsupported_terms(question)
    if unsupported_terms and not context_mentions_terms(context, unsupported_terms):
        return UNKNOWN_ANSWER, []

    from groq import Groq

    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    completion = client.chat.completions.create(
        model=model,
        temperature=0.1,
        max_completion_tokens=2048,
        top_p=1,
        reasoning_effort="medium",
        stream=False,
        stop=None,
        messages=[
            {
                "role": "system",
                "content": course_answer_style_prompt("retrieved course context"),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nRetrieved context:\n{context}",
            },
        ],
    )
    return format_model_answer(completion.choices[0].message.content), unique_sources(context_rows)



def answer_auto(question: str, ready: dict[str, bool]):
    try:
        response = answer_with_groq(question)
        return (
            response,
            f"Groq graph chat: {os.getenv('GROQ_MODEL', 'openai/gpt-oss-120b')}",
            [classify_question(question).replace("_", " ").title()],
        )
    except Exception:
        query_name = classify_question(question)
        return (
            format_direct_answer(query_name, run_query(query_name)),
            f"Direct graph answer: {query_name}",
            [query_name.replace("_", " ").title()],
        )



@app.get("/")
def index():
    return render_template("index.html", status=env_ready())


@app.get("/api/status")
def status():
    ready = env_ready()
    return jsonify(
        {
            "neo4j": ready["neo4j"],
            "openai": ready["openai"],
            "groq": ready["groq"],
            "vector": ready["vector"],
            "mode": (
                "Vector + Groq"
                if ready["groq"] and ready["vector"]
                else "Groq graph chat"
                if ready["groq"]
                else "Direct graph answers"
            ),
        }
    )


@app.post("/api/chat")
def chat():
    payload = request.get_json(force=True)
    original_question = str(payload.get("message", "")).strip()
    requested_mode = str(payload.get("mode", "auto"))
    if not original_question:
        return jsonify({"error": "Please enter a question."}), 400

    ready = env_ready()
    question = original_question
    if ready["groq"]:
        try:
            question = refine_user_question(original_question)
        except Exception:
            question = original_question

    try:
        sources: list[str] = []
        if requested_mode == "auto":
            response, mode, sources = answer_auto(question, ready)
        elif requested_mode == "vector":
            response, sources = answer_with_vector_groq(question)
            mode = f"Vector + Groq: {os.getenv('GROQ_MODEL', 'openai/gpt-oss-120b')}"
        elif requested_mode == "groq":
            response = answer_with_groq(question)
            mode = f"Groq graph chat: {os.getenv('GROQ_MODEL', 'openai/gpt-oss-120b')}"
            sources = [classify_question(question).replace("_", " ").title()]
        elif requested_mode == "llm":
            response = answer_with_llm(question)
            mode = "OpenAI vector chat"
        else:
            query_name = classify_question(question)
            response = format_direct_answer(query_name, run_query(query_name))
            mode = f"Direct graph answer: {query_name}"
            sources = [query_name.replace("_", " ").title()]
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify(
        {
            "answer": response,
            "mode": mode,
            "sources": sources,
            "rewritten_question": question if question != original_question else "",
        }
    )



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

