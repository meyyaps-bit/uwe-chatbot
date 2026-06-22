from __future__ import annotations

import os
import sys
import textwrap

from dotenv import load_dotenv
from langchain.chains import GraphCypherQAChain, RetrievalQAWithSourcesChain
from langchain.prompts.prompt import PromptTemplate
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


VECTOR_INDEX_NAME = "uwe_course_chunks"
VECTOR_SOURCE_PROPERTY = "text"
VECTOR_EMBEDDING_PROPERTY = "textEmbedding"


def get_graph() -> Neo4jGraph:
    load_dotenv(".env", override=True)
    return Neo4jGraph(
        url=os.environ["NEO4J_URI"],
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"],
        database=os.getenv("NEO4J_DATABASE") or "neo4j",
    )


def vector_chain():
    load_dotenv(".env", override=True)
    retrieval_query_window = """
    MATCH window=(:Chunk)-[:NEXT*0..1]->(node)-[:NEXT*0..1]->(:Chunk)
    WITH node, score, window AS longestWindow
      ORDER BY length(window) DESC LIMIT 1
    WITH nodes(longestWindow) AS chunkList, node, score
      UNWIND chunkList AS chunkRows
    WITH collect(chunkRows.text) AS textList, node, score
    RETURN apoc.text.join(textList, "\n\n") AS text,
      score,
      {source: node.source, section: node.sectionTitle} AS metadata
    """
    vector_store = Neo4jVector.from_existing_index(
        OpenAIEmbeddings(),
        url=os.environ["NEO4J_URI"],
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"],
        database=os.getenv("NEO4J_DATABASE") or "neo4j",
        index_name=VECTOR_INDEX_NAME,
        text_node_property=VECTOR_SOURCE_PROPERTY,
        embedding_node_property=VECTOR_EMBEDDING_PROPERTY,
        retrieval_query=retrieval_query_window,
    )
    return RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(temperature=0),
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
    )


def cypher_chain(graph: Neo4jGraph):
    template = """Task: Generate a Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use relationship types or properties that are not in the schema.
Do not include explanations. Return only the Cypher statement.

Schema:
{schema}

Examples:

# What are the core modules?
MATCH (:Course)-[:HAS_CORE_MODULE]->(module:Module)
RETURN module.name

# What are the optional modules?
MATCH (:Course)-[:HAS_OPTIONAL_MODULE]->(module:Module)
RETURN module.name

# What careers can graduates pursue?
MATCH (:Course)-[:LEADS_TO_ROLE]->(role:CareerRole)
RETURN role.name

# What is the course campus and duration?
MATCH (course:Course)
RETURN course.campus, course.duration

# How much is the international full-time fee?
MATCH (course:Course)-[:HAS_SECTION]->(section:Section)
WHERE section.sectionKey = "fees"
MATCH (section)-[:HAS_CHUNK]->(chunk:Chunk)
RETURN chunk.text

The question is:
{question}"""
    prompt = PromptTemplate(
        input_variables=["schema", "question"],
        template=template,
    )
    return GraphCypherQAChain.from_llm(
        ChatOpenAI(temperature=0),
        graph=graph,
        verbose=False,
        cypher_prompt=prompt,
    )


def answer(question: str, mode: str = "vector") -> str:
    if mode == "cypher":
        graph = get_graph()
        return cypher_chain(graph).run(question)
    response = vector_chain()({"question": question}, return_only_outputs=True)
    return response["answer"].strip()


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python3 chat_uwe_kg.py "your question" [vector|cypher]')
        raise SystemExit(2)

    question = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "vector"
    print(textwrap.fill(answer(question, mode=mode), width=88))


if __name__ == "__main__":
    main()

