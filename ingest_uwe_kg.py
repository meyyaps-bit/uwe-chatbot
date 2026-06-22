from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase
import ssl
import certifi


from embeddings import VECTOR_DIMENSIONS, embed_texts


DATA_FILE = Path("data/uwe_msc_data_science_complete_scrape.json")
RAW_TEXT_FILE = Path("data/uwe_msc_data_science_raw_main_text.txt")
VECTOR_INDEX_NAME = "uwe_course_chunks"


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def flatten_text(value: Any, prefix: str | None = None) -> list[str]:
    """Turn nested JSON values into readable text lines for chunking."""
    lines: list[str] = []
    label = prefix.replace("_", " ").title() if prefix else None

    if value is None:
        return lines
    if isinstance(value, str):
        if value.strip():
            lines.append(f"{label}: {value.strip()}" if label else value.strip())
        return lines
    if isinstance(value, (int, float)):
        lines.append(f"{label}: {value}" if label else str(value))
        return lines
    if isinstance(value, list):
        if label:
            lines.append(f"{label}:")
        for item in value:
            if isinstance(item, dict):
                lines.extend(flatten_text(item))
            else:
                item_lines = flatten_text(item)
                lines.extend(f"- {line}" for line in item_lines)
        return lines
    if isinstance(value, dict):
        if label:
            lines.append(f"{label}:")
        for key, nested in value.items():
            lines.extend(flatten_text(nested, key))
        return lines
    return lines


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 0) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""

    expanded: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= chunk_size:
            expanded.append(paragraph)
            continue
        words = paragraph.split()
        part = ""
        for word in words:
            if not part:
                part = word
            elif len(part) + len(word) + 1 <= chunk_size:
                part = f"{part} {word}"
            else:
                expanded.append(part)
                tail = part[-overlap:] if overlap and len(part) > overlap else ""
                part = f"{tail} {word}".strip() if tail else word
        if part:
            expanded.append(part)

    for paragraph in expanded:
        if not current:
            current = paragraph
        elif len(current) + len(paragraph) + 2 <= chunk_size:
            current = f"{current}\n\n{paragraph}"
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap and len(current) > overlap else ""
            current = f"{tail}\n\n{paragraph}" if tail else paragraph

    if current:
        chunks.append(current)
    return chunks


def build_section_records(data: dict[str, Any]) -> list[dict[str, Any]]:
    section_keys = [
        "overview",
        "about",
        "entry",
        "structure",
        "learning_and_teaching",
        "study_time",
        "assessment",
        "fees",
        "features",
        "careers",
        "life",
        "talk_to_a_lecturer",
        "contact_us",
    ]
    records: list[dict[str, Any]] = []
    for seq, key in enumerate(section_keys):
        if key not in data:
            continue
        title = key.replace("_", " ").title()
        text = "\n".join(flatten_text(data[key], key))
        if text.strip():
            records.append(
                {
                    "sectionId": f"uwe-msc-data-science-{slugify(key)}",
                    "sectionKey": key,
                    "title": title,
                    "seq": seq,
                    "text": text,
                }
            )
    return records


def assert_env() -> dict[str, str]:
    load_dotenv(".env", override=True)
    keys = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "NEO4J_DATABASE"]
    env = {key: os.getenv(key, "") for key in keys}
    missing = [key for key, value in env.items() if not value]
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")
    return env


def run_write(session, query: str, **params):
    return session.run(query, **params).consume()


def main() -> None:
    env = assert_env()
    data = json.loads(DATA_FILE.read_text())

    course_id = "uwe-msc-data-science-inb112"
    course = {
        "courseId": course_id,
        "title": data["page_title"],
        "institution": data["institution"],
        "sourceUrl": data["source_url"],
        "pageLastUpdated": data["page_last_updated"],
        "courseStatus": data["course_status"],
        **data["overview"],
    }
    course["programme_leaders"] = data["overview"].get("programme_leaders", [])
    course["accreditations_and_partnerships"] = data["overview"].get(
        "accreditations_and_partnerships", []
    )

    section_records = build_section_records(data)
    chunk_records: list[dict[str, Any]] = []
    for section in section_records:
        for seq, text in enumerate(chunk_text(section["text"])):
            chunk_records.append(
                {
                    "chunkId": f"{section['sectionId']}-chunk{seq:04d}",
                    "courseId": course_id,
                    "sectionId": section["sectionId"],
                    "sectionKey": section["sectionKey"],
                    "sectionTitle": section["title"],
                    "chunkSeqId": seq,
                    "text": text,
                    "source": data["source_url"],
                }
            )



    
    driver = GraphDatabase.driver(
        env["NEO4J_URI"],
        auth=(env["NEO4J_USERNAME"], env["NEO4J_PASSWORD"])
    )





    with driver.session(database=env["NEO4J_DATABASE"]) as session:
        run_write(
            session,
            """
            MATCH (chunk:Chunk {courseId: $courseId})
            DETACH DELETE chunk
            """,
            courseId=course_id,
        )
        run_write(
            session,
            """
            MATCH (section:Section)
            WHERE section.sectionId STARTS WITH $sectionPrefix
            DETACH DELETE section
            """,
            sectionPrefix=f"{course_id.rsplit('-', 1)[0]}-",
        )
        run_write(
            session,
            """
            MATCH (course:Course {courseId: $courseId})
            DETACH DELETE course
            """,
            courseId=course_id,
        )

        run_write(session, "CREATE CONSTRAINT uwe_course_id IF NOT EXISTS FOR (c:Course) REQUIRE c.courseId IS UNIQUE")
        run_write(session, "CREATE CONSTRAINT uwe_section_id IF NOT EXISTS FOR (s:Section) REQUIRE s.sectionId IS UNIQUE")
        run_write(session, "CREATE CONSTRAINT uwe_chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE")
        run_write(session, "CREATE CONSTRAINT uwe_module_name IF NOT EXISTS FOR (m:Module) REQUIRE m.name IS UNIQUE")
        run_write(session, "CREATE CONSTRAINT uwe_role_name IF NOT EXISTS FOR (r:CareerRole) REQUIRE r.name IS UNIQUE")
        run_write(session, "CREATE FULLTEXT INDEX uweSectionTitles IF NOT EXISTS FOR (s:Section) ON EACH [s.title, s.sectionKey]")
        run_write(session, "CREATE FULLTEXT INDEX uweModuleNames IF NOT EXISTS FOR (m:Module) ON EACH [m.name]")
        run_write(session, f"DROP INDEX {VECTOR_INDEX_NAME} IF EXISTS")

        run_write(
            session,
            f"""
            CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS
            FOR (c:Chunk) ON (c.textEmbedding)
            OPTIONS {{ indexConfig: {{
              `vector.dimensions`: {VECTOR_DIMENSIONS},
              `vector.similarity_function`: 'cosine'
            }}}}
            """,
        )

        run_write(
            session,
            """
            MERGE (course:Course {courseId: $course.courseId})
            SET course += $course
            """,
            course=course,
        )

        for section in section_records:
            run_write(
                session,
                """
                MATCH (course:Course {courseId: $courseId})
                MERGE (section:Section {sectionId: $section.sectionId})
                SET section += $section
                MERGE (course)-[:HAS_SECTION]->(section)
                """,
                courseId=course_id,
                section=section,
            )

        for chunk in chunk_records:
            run_write(
                session,
                """
                MATCH (section:Section {sectionId: $chunk.sectionId})
                MERGE (chunk:Chunk {chunkId: $chunk.chunkId})
                SET chunk += $chunk
                MERGE (section)-[:HAS_CHUNK]->(chunk)
                MERGE (chunk)-[:PART_OF]->(section)
                """,
                chunk=chunk,
            )

        for section in section_records:
            run_write(
                session,
                """
                MATCH (chunk:Chunk {sectionId: $sectionId})
                WITH chunk ORDER BY chunk.chunkSeqId ASC
                WITH collect(chunk) AS chunks
                FOREACH (i IN range(0, size(chunks) - 2) |
                  FOREACH (from IN [chunks[i]] |
                    FOREACH (to IN [chunks[i + 1]] |
                      MERGE (from)-[:NEXT]->(to)
                    )
                  )
                )
                """,
                sectionId=section["sectionId"],
            )

        structure = data.get("structure", {})
        for module_name in structure.get("core_modules", []):
            run_write(
                session,
                """
                MATCH (course:Course {courseId: $courseId})
                MERGE (module:Module {name: $name})
                SET module.kind = 'core'
                MERGE (course)-[:HAS_CORE_MODULE]->(module)
                """,
                courseId=course_id,
                name=module_name,
            )
        for module_name in structure.get("optional_modules", []):
            run_write(
                session,
                """
                MATCH (course:Course {courseId: $courseId})
                MERGE (module:Module {name: $name})
                SET module.kind = 'optional'
                MERGE (course)-[:HAS_OPTIONAL_MODULE]->(module)
                """,
                courseId=course_id,
                name=module_name,
            )

        for role in data.get("careers", {}).get("career_roles", []):
            run_write(
                session,
                """
                MATCH (course:Course {courseId: $courseId})
                MERGE (role:CareerRole {name: $name})
                MERGE (course)-[:LEADS_TO_ROLE]->(role)
                """,
                courseId=course_id,
                name=role,
            )

        contact = data.get("contact_us", {})
        if contact:
            run_write(
                session,
                """
                MATCH (course:Course {courseId: $courseId})
                MERGE (contact:Contact {institution: $contact.institution})
                SET contact += $contact
                MERGE (course)-[:CONTACT]->(contact)
                """,
                courseId=course_id,
                contact=contact,
            )

        if chunk_records:
            vectors = embed_texts([chunk["text"] for chunk in chunk_records])
            for chunk, vector in zip(chunk_records, vectors):
                session.run(
                    """
                    MATCH (chunk:Chunk {chunkId: $chunkId})
                    CALL db.create.setNodeVectorProperty(chunk, "textEmbedding", $vector)
                    """,
                    chunkId=chunk["chunkId"],
                    vector=vector,
                ).consume()
            embedding_status = f"local MiniLM embeddings created ({VECTOR_DIMENSIONS} dimensions)"
        else:
            embedding_status = "no chunks available for embedding"

        vector_count = session.run(
            """
            MATCH (chunk:Chunk {courseId: $courseId})
            WHERE chunk.textEmbedding IS NOT NULL
            RETURN count(chunk) AS count
            """,
            courseId=course_id,
        ).single()["count"]

        if vector_count != len(chunk_records):
            raise RuntimeError(
                f"Expected {len(chunk_records)} embedded chunks, found {vector_count}"
            )

        counts = session.run(
            """
            MATCH (n)
            WHERE n:Course OR n:Section OR n:Chunk OR n:Module OR n:CareerRole OR n:Contact
            RETURN labels(n)[0] AS label, count(*) AS count
            ORDER BY label
            """
        ).data()

    driver.close()
    print("Ingest complete.")
    print(f"Sections: {len(section_records)}")
    print(f"Chunks: {len(chunk_records)}")
    print(f"Embedding status: {embedding_status}")
    print("Counts:")
    for row in counts:
        print(f"  {row['label']}: {row['count']}")


if __name__ == "__main__":
    main()
