from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from neo4j import GraphDatabase


QUERIES = {
    "summary": """
        MATCH (course:Course)
        RETURN course.title AS title,
               course.institution AS institution,
               course.course_code AS code,
               course.campus AS campus,
               course.duration AS duration,
               course.courseStatus AS status
    """,
    "modules": """
        MATCH (:Course)-[rel]->(module:Module)
        WHERE type(rel) IN ["HAS_CORE_MODULE", "HAS_OPTIONAL_MODULE"]
        RETURN type(rel) AS moduleType, module.name AS module
        ORDER BY moduleType, module
    """,
    "careers": """
        MATCH (:Course)-[:LEADS_TO_ROLE]->(role:CareerRole)
        RETURN role.name AS careerRole
        ORDER BY careerRole
    """,
    "fees": """
        MATCH (:Course)-[:HAS_SECTION]->(:Section {sectionKey: "fees"})-[:HAS_CHUNK]->(chunk:Chunk)
        RETURN chunk.text AS fees
        ORDER BY chunk.chunkSeqId
    """,
    "entry": """
        MATCH (:Course)-[:HAS_SECTION]->(:Section {sectionKey: "entry"})-[:HAS_CHUNK]->(chunk:Chunk)
        RETURN chunk.text AS entryRequirements
        ORDER BY chunk.chunkSeqId
    """,
    "counts": """
        MATCH (n)
        WHERE n:Course OR n:Section OR n:Chunk OR n:Module OR n:CareerRole OR n:Contact
        RETURN labels(n)[0] AS label, count(*) AS count
        ORDER BY label
    """,
}


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in QUERIES:
        names = ", ".join(sorted(QUERIES))
        print(f"Usage: python3 query_uwe_kg.py <{names}>")
        raise SystemExit(2)

    load_dotenv(".env", override=True)
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    with driver.session(database=os.getenv("NEO4J_DATABASE") or "neo4j") as session:
        rows = session.run(QUERIES[sys.argv[1]]).data()
    driver.close()

    for row in rows:
        print(row)


if __name__ == "__main__":
    main()

