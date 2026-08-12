from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from app.cache.simple_cache import SimpleCache

load_dotenv()
rewrite_cache = SimpleCache()

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0,)


rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You rewrite user questions into concise semantic-search queries
for a job-opportunity evidence retrieval system.

The search corpus may contain:
- candidate skills
- candidate projects
- candidate experience
- job requirements
- job responsibilities
- company and recruitment evidence

RULES:

1. Preserve the original information need.
2. Add useful terminology and closely related concepts when helpful.
3. Do not answer the question.
4. Do not invent candidate, job, or company facts.
5. Do not make the query unnecessarily broad.
6. Return only the rewritten retrieval query.
        """
    ),
    (
        "user",
        "{query}"
    ),
])


query_rewrite_chain = (rewrite_prompt | model)


def rewrite_query(query: str) -> str:
    cached = rewrite_cache.get(query)

    if cached is not None:
        print("[CACHE HIT] query rewrite")
        return cached

    print("[CACHE MISS] query rewrite")

    response = query_rewrite_chain.invoke({"query": query})

    text_parts = []

    for block in response.content_blocks:
        if block.get("type") == "text":
            text = block.get("text")

            if text:
                text_parts.append(text)

    rewritten_query = " ".join(text_parts).strip()

    if not rewritten_query:
        raise ValueError("Query rewriter returned no text content.")

    rewrite_cache.set(query, rewritten_query)

    return rewritten_query