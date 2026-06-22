# UWE MSc Data Science Knowledge Graph RAG

This project builds a Flask chat interface for questions about the UWE Bristol MSc Data Science course. It uses course data from `data/`, ingests the content into Neo4j as a knowledge graph, adds local MiniLM embeddings for semantic retrieval, and can use Groq for source-grounded answer generation.

The app supports direct graph queries for common course questions and an `Auto` chat mode that can combine structured graph facts with vector retrieval.

## Architecture

Current data sources:

- `data/uwe_msc_data_science_complete_scrape.json`
- `data/uwe_msc_data_science_raw_main_text.txt`

Main components:

- `ingest_uwe_kg.py` loads the scraped course data, creates graph nodes and relationships, chunks course sections, stores embeddings, and creates the Neo4j vector index.
- `query_uwe_kg.py` runs direct Cypher checks for summary, modules, fees, entry requirements, careers, and graph counts.
- `app.py` serves the Flask web app and exposes chat/status API endpoints.
- `static/app.js` handles the browser chat workflow, quick questions, status display, and API calls.
- `templates/index.html` and `static/styles.css` provide the web chat interface.

## Graph Model

Nodes:

- `(:Course)` for the MSc Data Science course
- `(:Section)` for page sections such as overview, entry, structure, fees, and careers
- `(:Chunk)` for searchable text chunks
- `(:Module)` for core and optional modules
- `(:CareerRole)` for listed career outcomes
- `(:Contact)` for institution contact details

Relationships:

- `(course)-[:HAS_SECTION]->(section)`
- `(section)-[:HAS_CHUNK]->(chunk)`
- `(chunk)-[:PART_OF]->(section)`
- `(chunk)-[:NEXT]->(chunk)` for ordered context windows
- `(course)-[:HAS_CORE_MODULE]->(module)`
- `(course)-[:HAS_OPTIONAL_MODULE]->(module)`
- `(course)-[:LEADS_TO_ROLE]->(careerRole)`
- `(course)-[:CONTACT]->(contact)`

## Setup

1. Create your local `.env`.

Copy the example file and replace the placeholders with your own Neo4j and API credentials:

```bash
cp .env.example .env
```

```bash
NEO4J_URI=replace_with_your_neo4j_uri
NEO4J_USERNAME=replace_with_your_neo4j_username
NEO4J_PASSWORD=replace_with_your_neo4j_password
NEO4J_DATABASE=replace_with_your_neo4j_database
OPENAI_API_KEY=replace_with_your_openai_api_key
GROQ_API_KEY=replace_with_your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
```

Do not commit `.env`, `Neo4j-password.txt`, or any file containing real credentials. The active web chat path can use Groq for answer generation. OpenAI is no longer required for the active embedding path.

2. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

3. Ingest the current data into Neo4j:

```bash
python3 ingest_uwe_kg.py
```

4. Query the graph without an LLM:

```bash
python3 query_uwe_kg.py summary
python3 query_uwe_kg.py modules
python3 query_uwe_kg.py fees
python3 query_uwe_kg.py entry
python3 query_uwe_kg.py careers
python3 query_uwe_kg.py counts
```

5. Run the web chat interface:

```bash
python3 app.py
```

Open:

```text
http://127.0.0.1:5000
```

The interface answers from direct graph queries immediately. After `python3 ingest_uwe_kg.py` creates local MiniLM embeddings and a real `GROQ_API_KEY` is configured, `Auto` mode can use vector retrieval plus Groq answer generation.

## Project Plan Workbook

The project plan workbook is based on `Activity Template_ Project Plan.xlsx`. The filled workbook documents a four-week implementation plan for this chatbot project.

Planned workbook tabs:

- `Tasks and Timeline`: dated implementation schedule and Gantt-style task tracker
- `Task Brainstorm`: grouped task candidates and milestone ideas
- `Additional Resources`: project files, reference documentation, and useful links
- `Quality and Evaluation`: acceptance criteria and evaluation indicators
- `Survey Questions`: user testing questions for chatbot quality and usability

## What Was Applied From The Lessons

- Lesson 3: vector index and local MiniLM embedding-based semantic search
- Lesson 4: split source text into chunks and store them as graph nodes
- Lesson 5: link chunks with `NEXT` and use window retrieval
- Lesson 6: add structured entities and graph relationships, then augment retrieval with graph facts
- Lesson 7: include a Cypher query mode for structured questions
