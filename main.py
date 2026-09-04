import os
import time
import json
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from openai import OpenAI

import rag

app = FastAPI(
    title="Week 15 AI Fellowship Assistant API",
    description="Backend API for AI assistant, RAG, tool calling, and fallback handling."
)

# read settings from environment or use local defaults
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "dummy-key")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "gpt-4o-mini")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gpt-3.5-turbo")
USE_VLLM = os.getenv("USE_VLLM", "false").lower() == "true"

# initialize openai client (can point to local vllm if enabled)
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=VLLM_BASE_URL if USE_VLLM else None
)


# in-memory rate limiting: keep track of request timestamps per ip
request_timestamps = {}
RATE_LIMIT_MAX = 60
RATE_LIMIT_WINDOW = 60  # seconds


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # skip rate limiting for health check and docs
    if request.url.path in ["/health", "/", "/docs", "/openapi.json"]:
        return await call_next(request)

    client_ip = request.client.host if request.client else "127.0.0.1"
    now = time.time()

    # filter timestamps within the last window
    history = [t for t in request_timestamps.get(client_ip, []) if now - t < RATE_LIMIT_WINDOW]
    if len(history) >= RATE_LIMIT_MAX:
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests. Please slow down and try again in a minute."}
        )

    history.append(now)
    request_timestamps[client_ip] = history

    return await call_next(request)


# request and response models
class ChatRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = "You are a helpful AI assistant."
    model: Optional[str] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    use_tool: Optional[bool] = False
    use_structured_output: Optional[bool] = False


class StudentEvaluation(BaseModel):
    student_name: str = Field(description="Name of the student or user")
    summary: str = Field(description="Brief summary of their question or submission")
    score: int = Field(description="Score between 0 and 100")
    passed: bool = Field(description="Whether the evaluation passed or failed")


class RagRequest(BaseModel):
    question: str
    temperature: Optional[float] = 0.5
    top_p: Optional[float] = 1.0


# dummy weather function for tool calling
def get_weather(location: str):
    loc = location.lower()
    if "tokyo" in loc:
        return json.dumps({"location": "Tokyo", "temperature": "18C", "condition": "Sunny"})
    elif "paris" in loc:
        return json.dumps({"location": "Paris", "temperature": "22C", "condition": "Cloudy"})
    elif "kathmandu" in loc:
        return json.dumps({"location": "Kathmandu", "temperature": "24C", "condition": "Pleasant"})
    else:
        return json.dumps({"location": location, "temperature": "20C", "condition": "Partly Cloudy"})


weather_tool_schema = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a given city or location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, e.g. Kathmandu, Tokyo, London"
                }
            },
            "required": ["location"]
        }
    }
}


def run_llm_with_fallback(messages, model=None, temperature=0.7, top_p=1.0, tools=None, response_format=None):
    # pick model
    target_model = model or PRIMARY_MODEL

    # attempt 1: try primary model
    try:
        kwargs = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p
        }
        if tools:
            kwargs["tools"] = tools
        if response_format:
            kwargs["response_format"] = response_format

        resp = client.chat.completions.create(**kwargs)
        return resp, target_model

    except Exception as err1:
        print(f"Primary model {target_model} failed: {err1}. Retrying with fallback {FALLBACK_MODEL}...")

        # attempt 2: retry with fallback model
        try:
            kwargs = {
                "model": FALLBACK_MODEL,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p
            }
            if tools:
                kwargs["tools"] = tools
            if response_format:
                kwargs["response_format"] = response_format

            resp = client.chat.completions.create(**kwargs)
            return resp, FALLBACK_MODEL

        except Exception as err2:
            print(f"Fallback model failed too: {err2}. Falling back to offline mock response.")
            return None, "offline-mode"


@app.get("/")
async def index():
    return {
        "message": "Week 15 backend running",
        "routes": ["/chat", "/rag-chat", "/ingest", "/health"]
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest_docs(folder_path: str = "./docs"):
    try:
        count = rag.ingest_documents(folder_path)
        return {"status": "success", "chunks_ingested": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(req: ChatRequest):
    messages = [
        {"role": "system", "content": req.system_prompt},
        {"role": "user", "content": req.prompt}
    ]

    tools = [weather_tool_schema] if req.use_tool else None
    response_format = {"type": "json_object"} if req.use_structured_output else None

    # prompt guidance if structured json is needed
    if req.use_structured_output:
        schema_json = json.dumps(StudentEvaluation.model_json_schema())
        messages[0]["content"] += f"\nReturn valid JSON adhering to this schema: {schema_json}"

    response, used_model = run_llm_with_fallback(
        messages=messages,
        model=req.model,
        temperature=req.temperature,
        top_p=req.top_p,
        tools=tools,
        response_format=response_format
    )

    # offline mock handler (when testing without API key or vLLM server)
    if response is None:
        if req.use_tool and "weather" in req.prompt.lower():
            res = get_weather("Kathmandu")
            return {
                "model": "offline-mode",
                "content": f"Simulated weather result: {res}",
                "tool_called": "get_weather",
                "tool_args": {"location": "Kathmandu"},
                "tool_result": json.loads(res)
            }
        elif req.use_structured_output:
            sample = StudentEvaluation(
                student_name="Fellowship Student",
                summary=req.prompt,
                score=92,
                passed=True
            ).model_dump()
            return {
                "model": "offline-mode",
                "content": json.dumps(sample),
                "structured_data": sample
            }
        else:
            return {
                "model": "offline-mode",
                "content": f"[Offline response] Query: '{req.prompt}'. API key / vLLM was not available, but server and flow work."
            }

    msg = response.choices[0].message

    # handle tool call if returned by model
    if msg.tool_calls:
        call = msg.tool_calls[0]
        name = call.function.name
        args = json.loads(call.function.arguments)

        if name == "get_weather":
            city = args.get("location", "Kathmandu")
            tool_output = get_weather(city)
        else:
            tool_output = json.dumps({"error": "Unknown tool"})

        return {
            "model": used_model,
            "tool_called": name,
            "tool_args": args,
            "tool_result": json.loads(tool_output),
            "content": f"Executed tool '{name}': {tool_output}"
        }

    structured_data = None
    if req.use_structured_output:
        try:
            structured_data = json.loads(msg.content)
        except Exception:
            structured_data = {"raw": msg.content}

    return {
        "model": used_model,
        "content": msg.content,
        "structured_data": structured_data
    }


@app.post("/rag-chat")
async def rag_chat(req: RagRequest):
    # retrieve chunks from vector store
    chunks = rag.query_rag(req.question, n_results=3)

    if chunks:
        context_lines = []
        for c in chunks:
            src = c.get("metadata", {}).get("source", "docs")
            context_lines.append(f"[{src}]: {c['text']}")
        context_str = "\n\n".join(context_lines)
    else:
        context_str = "No relevant context found."

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the question."},
        {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {req.question}\nAnswer:"}
    ]

    response, used_model = run_llm_with_fallback(
        messages=messages,
        temperature=req.temperature,
        top_p=req.top_p
    )

    if response is None:
        answer = f"[Offline RAG Answer] Retrieved {len(chunks)} chunks from ChromaDB for question: {req.question}"
    else:
        answer = response.choices[0].message.content

    return {
        "model": used_model,
        "question": req.question,
        "answer": answer,
        "retrieved_context": chunks
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
