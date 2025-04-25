import requests
from typing import Any
from multi_agent_orchestrator.types import ConversationMessage, ParticipantRole

# Tool description for Claude
job_search_tool_description = [{
    "toolSpec": {
        "name": "Job_Search_Tool",
        "description": "Search for job listings based on user input like title, location, experience, or role.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Job search query with role, location, or experience (e.g., 'AI engineer in Singapore')",
                    }
                },
                "required": ["query"],
            }
        },
    }
}]

job_search_tool_prompt = """
You are a career assistant who helps users find job listings. Use only the Job_Search_Tool for your responses.
- Ask the user for a clear query like "AI engineer in Singapore" or "chef in Vienna"
- Use the tool by passing that full query to the input.
- Do not make up jobs or fabricate results.
- Explain steps to the user like "Searching for jobs...", then show the result list.
- Limit results to top 5 matching listings.
- Stick to actual job portals like LinkedIn, Indeed, or Glassdoor.
"""

SERPER_API_KEY = "your-api-key"  # or use `os.getenv("SERPER_API_KEY")`

SERPER_API_URL = "https://google.serper.dev/search"


async def job_search_tool_handler(response: ConversationMessage, conversation: list[dict[str, Any]]) -> ConversationMessage:
    results = []

    for content_block in response.content:
        if "toolUse" in content_block:
            tool_use = content_block["toolUse"]
            if tool_use["name"] == "Job_Search_Tool":
                tool_response = await search_jobs(tool_use["input"])
                results.append({
                    "toolResult": {
                        "toolUseId": tool_use["toolUseId"],
                        "content": [{"json": {"results": tool_response}}],
                    }
                })

    return ConversationMessage(
        role=ParticipantRole.USER.value,
        content=results
    )


async def search_jobs(input_data: dict[str, str]) -> list[dict[str, str]]:
    query = input_data.get("query", "")
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {"q": query, "gl": "us", "hl": "en"}

    try:
        response = requests.post(SERPER_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json().get("organic", [])

        allowed_domains = ["linkedin.com", "indeed.com", "glassdoor.com", "monster.com"]

        jobs = [
            {
                "title": r.get("title"),
                "link": r.get("link"),
                "description": r.get("snippet"),
                "source": r.get("source", "Job Portal")
            }
            for r in data if any(domain in r.get("link", "") for domain in allowed_domains)
        ]
        return jobs[:5]

    except Exception as e:
        return [{"error": str(e)}]
