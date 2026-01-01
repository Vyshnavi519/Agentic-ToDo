from __future__ import annotations
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Callable, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types

@dataclass
class TodoItem:
    task: str
    status: str  

class TodoStore: # All reads/writes go through this class to avoid state drift.
    # This class encapsulates in-memory todo state and persistence(this is extra).
    def __init__(self, state_file: str = "todos.json") -> None:
        self.state_file = state_file
        self._todos: List[TodoItem] = []

    @staticmethod
    def normalize_task(s: str) -> str: # Normalization to make sure that the duplicate is identifed consistenly
        return " ".join(s.strip().split()).lower()

    def load(self) -> None:
        if not os.path.exists(self.state_file):
            self._todos = []
            return
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            items: List[TodoItem] = []
            if isinstance(data, list):
                for x in data:
                    if isinstance(x, str):
                        items.append(TodoItem(task=x, status="open"))
                    elif isinstance(x, dict) and "task" in x:
                        items.append(
                            TodoItem(
                                task=str(x["task"]),
                                status=str(x.get("status", "open")),
                            )
                        )
            self._todos = items
        except Exception:
            self._todos = []

    def save(self) -> None:
        payload = [{"task": t.task, "status": t.status} for t in self._todos]
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
# below are the four tool call methods from the assessment
    def add_todo(self, item: str) -> Dict[str, Any]: #add_todo tool
        item_n = self.normalize_task(item)
        if not item_n:
            return {"ok": False, "error": "Error: Item cannot be empty"}
        for t in self._todos:
            if self.normalize_task(t.task) == item_n:
                return {"ok": False, "error": "Error: Item already exists", "item": item_n}

        display_task = " ".join(item.strip().split())
        self._todos.append(TodoItem(task=display_task, status="open"))
        self.save()
        return {"ok": True, "added": {"task": display_task, "status": "open"}}

    # complete_todo
    def complete_todo(self, item: str) -> Dict[str, Any]:
        item_n = self.normalize_task(item)
        if not item_n:
            return {"ok": False, "error": "Error: Item cannot be empty"}

        for t in self._todos:
            if self.normalize_task(t.task) == item_n:
                t.status = "done"
                self.save()
                return {"ok": True, "completed": {"task": t.task, "status": t.status}}

        return {"ok": False, "error": "Error: Item not found", "item": item_n}

# raw data display:
    def list_todos(self) -> List[Dict[str, str]]:
        return [{"task": t.task, "status": t.status} for t in self._todos]
# user list formatted display function:
    def format_list(self) -> str:
        if not self._todos:
            return "(empty)"
        lines = []
        for t in self._todos:
            box = "x" if t.status == "done" else " "
            lines.append(f"- [{box}] {t.task}")
        return "\n".join(lines)

class TodoAgent:
    """
    Encapsulates tool declarations, tool dispatch, conversation history (session),
    deterministic tool-result summarization, and retry/backoff for quota/rate limits.
    """
    def __init__(self, store: TodoStore, model: str) -> None:
        self.store = store
        self.model = model
        self.client = genai.Client()

        # Session history
        self.history: List[types.Content] = [
            self.content_text(self.system_instructions())
        ]
        self._tool_declarations = self.tool_declarations()
        self.tool_map: Dict[str, Callable[..., Any]] = {
            "add_todo": self.store.add_todo,
            "complete_todo": self.store.complete_todo,
            "list_todos": self.store.list_todos,
            "format_list": self.store.format_list,
        }
        self.config = self.build_config()
# Observability (extra)
    @staticmethod
    def trace(msg: str) -> None:
        print(f"[MindTrace] {msg}")
    def for_retry(
        self,
        *,
        contents: List[types.Content],
        max_retries: int = 5,
        base_delay_s: float = 1.0,
    ) -> types.GenerateContentResponse:
        """Retries transient quota/rate-limit errors with exponential backoff."""
        delay = base_delay_s
        for attempt in range(1, max_retries + 1):
            try:
                return self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=self.config,
                )
            except Exception as e:
                msg = str(e).lower()
                is_rate_limit = (
                    "429" in msg
                    or "resource_exhausted" in msg
                    or "rate limit" in msg
                    or "quota" in msg
                    or "too many requests" in msg
                )
                if not is_rate_limit or attempt == max_retries:
                    raise
                self.trace(
                    f"Rate limit/quota hit. Retrying in {delay:.1f}s "
                    f"(attempt {attempt}/{max_retries})"
                )
                time.sleep(delay)
                delay = min(delay * 2, 16.0)

    # System Instructions
    @staticmethod
    def system_instructions() -> str:
        return """You are a stateful To-Do assistant.
You have four tools: add_todo, complete_todo, list_todos, format_list.

CRITICAL RULES:
- If the user asks for JSON/raw data, call list_todos and present the raw array/object (no formatting).
- If the user asks to “show my list”, call format_list.
- For add requests, call add_todo once per item (multiple calls allowed).
- If add_todo returns an error about duplicates, tell the user the item already exists.
- Never fake updates: always use tools to read/modify the list.
Keep responses concise and helpful.
"""
# Tool Schemas:
    @staticmethod
    def tool_declarations() -> List[Dict[str, Any]]:
        return [
            {
                "name": "add_todo",
                "description": "Add a single todo item to the list. Must not add duplicates.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "item": {"type": "string", "description": "The todo text, e.g. 'Buy Milk'."}
                    },
                    "required": ["item"],
                },
            }, {
                "name": "complete_todo",
                "description": "Mark a todo item as completed using full string comparison.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "item": {"type": "string", "description": "Exact todo text to complete."}
                    },
                    "required": ["item"],
                },
            },
            {
                "name": "list_todos",
                "description": "Return the raw todo list as an array of objects (no formatting).",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "format_list",
                "description": "Return a formatted markdown-like checklist string for display.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        ]

    def build_config(self) -> types.GenerateContentConfig:
        tools = types.Tool(function_declarations=self._tool_declarations)
        return types.GenerateContentConfig(tools=[tools])

    @staticmethod
    def content_user(text: str) -> types.Content:
        return types.Content(role="user", parts=[types.Part(text=text)])

    @staticmethod
    def content_text(text: str) -> types.Content:
        return types.Content(role="model", parts=[types.Part(text=text)])

    @staticmethod
    def content_function_response(name: str, response_obj: Any) -> types.Content:
        return types.Content(
            role="tool",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        name=name,
                        response={"result": response_obj},
                    )
                )
            ],
        )

    @staticmethod
    def extract_function_calls(resp: types.GenerateContentResponse) -> List[types.FunctionCall]:
        calls: List[types.FunctionCall] = []
        if not resp.candidates:
            return calls
        content = resp.candidates[0].content
        if not content or not content.parts:
            return calls
        for p in content.parts:
            if p.function_call:
                calls.append(p.function_call)
        return calls

    @staticmethod
    def extract_text(resp: types.GenerateContentResponse) -> str:
        if getattr(resp, "text", None):
            return resp.text
        if not resp.candidates:
            return ""
        c = resp.candidates[0].content
        if not c or not c.parts:
            return ""
        chunks = []
        for p in c.parts:
            if p.text:
                chunks.append(p.text)
        return "".join(chunks).strip()

    @staticmethod
    def summarize(tool_results: List[Tuple[str, Any]]) -> Optional[str]:
        added: List[str] = []
        dupes: List[str] = []
        completed: List[str] = []
        not_found: List[str] = []
        for name, res in tool_results:
            if name == "add_todo":
                if isinstance(res, dict) and res.get("ok"):
                    added.append(res["added"]["task"])
                elif isinstance(res, dict) and res.get("error") == "Error: Item already exists":
                    dupes.append(res.get("item", ""))
            elif name == "complete_todo":
                if isinstance(res, dict) and res.get("ok"):
                    completed.append(res["completed"]["task"])
                elif isinstance(res, dict) and res.get("error") == "Error: Item not found":
                    not_found.append(res.get("item", ""))
        messages: List[str] = []
        if added:
            if len(added) == 1:
                messages.append(f'Added "{added[0]}" to your list.')
            else:
                joined = ", ".join(f'"{x}"' for x in added[:-1]) + f' and "{added[-1]}"'
                messages.append(f"Added {joined} to your list.")
        if dupes:
            if len(dupes) == 1:
                messages.append(f'You’ve already added "{dupes[0]}". It’s already on your list.')
            else:
                joined = ", ".join(f'"{x}"' for x in dupes[:-1]) + f' and "{dupes[-1]}"'
                messages.append(f"You’ve already added {joined}. They’re already on your list.")
        if completed:
            if len(completed) == 1:
                messages.append(f'Marked "{completed[0]}" as completed.')
            else:
                joined = ", ".join(f'"{x}"' for x in completed[:-1]) + f' and "{completed[-1]}"'
                messages.append(f"Marked {joined} as completed.")
        if not_found:
            if len(not_found) == 1:
                messages.append(f'I couldn’t find "{not_found[0]}" in your list.')
            else:
                joined = ", ".join(f'"{x}"' for x in not_found[:-1]) + f' and "{not_found[-1]}"'
                messages.append(f"I couldn’t find {joined} in your list.")
        return "\n".join(messages) if messages else None

    def turn(self, user_text: str) -> str:
        self.history = self.history + [self.content_user(user_text)]
        tool_results: List[Tuple[str, Any]] = []

        while True:
            resp = self.for_retry(contents=self.history)
            calls = self.extract_function_calls(resp)
            if calls:
                self.trace(f"Model requested {len(calls)} tool call(s).")
                self.history.append(resp.candidates[0].content)

                for fc in calls:
                    fn_name = fc.name
                    fn_args = dict(fc.args or {})
                    self.trace(f"Tool call → {fn_name}({fn_args})")
                    fn = self.tool_map.get(fn_name)
                    if not fn:
                        tool_result = {"ok": False, "error": f"Unknown tool: {fn_name}"}
                    else:
                        try:
                            tool_result = fn(**fn_args) if fn_args else fn()
                        except Exception as e:
                            tool_result = {"ok": False, "error": str(e)}
                    tool_results.append((fn_name, tool_result))
                    self.trace(f"Tool result ← {fn_name}: {tool_result}")
                    self.history.append(self.content_function_response(fn_name, tool_result))
                continue
# No more tool calls: model returned text
            self.history.append(resp.candidates[0].content)
# Prefer deterministic summary if tools were used
            if tool_results:
                summary = self.summarize(tool_results)
                if summary:
                    return summary
            return self.extract_text(resp)

def print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))
def wants_raw_json(user_text: str) -> bool:
    t = user_text.lower()
    return any(k in t for k in ["raw data","raw json","in json","as json","json output","raw strings",])
def wants_formatted_only(user_text: str) -> bool:
    t = user_text.lower()
    return any(k in t for k in ["show my list","my list","show list","formatted list","show tasks","full list","my todo list","todo list","my checklist",])
def wants_both(user_text: str) -> bool:
    t = user_text.lower()
    wants_raw = wants_raw_json(user_text)
    wants_formatted = any(k in t for k in ["my list","formatted","checklist","full list","formatted list","my checklist",])
    return wants_raw and wants_formatted
def main() -> None:
    load_dotenv()
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Missing GEMINI_API_KEY. Put it in your environment or .env file.")
        sys.exit(1)
    store = TodoStore(state_file="todos.json")
    store.load()
    agent = TodoAgent(store=store, model=model)
    print("Agentic To-Do. say exit or quit to stop.\n")
    while True:
        try:
            user_text = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            print("Finished")
            break
        TodoAgent.trace(f"User said: {user_text}")
        try: # if user asks for both raw data and formatted data or only raw data or only formatted data
            if wants_both(user_text):
                print("\nSystem (formatted list):")
                print(store.format_list())
                print("\nSystem (raw JSON):")
                print_json(store.list_todos())
                print()
                continue
            if wants_raw_json(user_text):
                print("\nSystem: Here is your raw data")
                print_json(store.list_todos())
                print()
                continue
            if wants_formatted_only(user_text):
                print("\nSystem: Here is your todo list of tasks")
                print(store.format_list())
                print()
                continue
            answer = agent.turn(user_text)
            print(f"\nSystem: {answer}\n")
        except Exception as e:
            print(f"\nSystem: Sorry something went wrong: {e}\n")
if __name__ == "__main__":
    main()
