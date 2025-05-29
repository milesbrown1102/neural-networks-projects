# llm_interface.py
import ollama
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
load_dotenv()

# Connect to Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def run_query(query):
    try:
        with driver.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]
    except Exception as e:
        return f"❌ Error running query: {e}"

def ask_llm(question, context=""):
    response = ollama.chat(
        model="mistral",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an educational assistant helping analyze student performance using a Neo4j database. "
                    "Only return Cypher queries that are executable — do not include placeholders like <value>. "
                    "If a specific value is unknown, make a reasonable assumption (e.g., passing score is 60)."
                )
            },
            {
                "role": "user",
                "content": f"{context}\n\nQuestion: {question}"
            }
        ]
    )

    llm_answer = response['message']['content']

    # Try to extract the Cypher query block
    if "```cypher" in llm_answer:
        try:
            cypher_code = llm_answer.split("```cypher")[1].split("```")[0].strip()
            # Fix common issues from LLM
            cypher_code = cypher_code.replace("quiz.score", "t.score")
            cypher_code = cypher_code.replace("relationshipCount", "size")
            query_result = run_query(cypher_code)
            return f"🤖 LLM Query:\n{cypher_code}\n\n📊 Query Result:\n{query_result}"
        except Exception as e:
            return f"🤖 LLM Response:\n{llm_answer}\n\n⚠️ Failed to parse or run Cypher query: {e}"
    else:
        return f"🤖 LLM Response:\n{llm_answer}\n\n⚠️ No Cypher query found."
