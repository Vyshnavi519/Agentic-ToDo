# Agentic To-Do 

This project implements a **agentic To-Do assistant** using the **Google Gen AI SDK (Gemini)**.

---

## Features

- To-Do Agent
- Tool-driven architecture:
  - `add_todo`
  - `complete_todo`
  - `list_todos` (raw data only)
  - `format_list` (user facing display)
- Duplicate detection handled inside tools
- Deterministic summaries of tool execution
- Optional local persistence via JSON file
- Basic observability through tool-call logging

---

## Requirements

- Python **3.9+**
- A Google Gen AI (Gemini) API key
---

## Installation

1. Clone the repository

```bash
git clone https://github.com/Vyshnavi519/Agentic-ToDo.git
cd agentic-todo
```
2. Create and activate a virtual environment 
create & activate
``` bash
python -m venv .venv
.venv\Scripts\activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Configure environment variables
    Create a .env file in the project root:
    GEMINI_API_KEY=your_api_key_here
    GEMINI_MODEL=gemini-2.5-flash

5. Start the agent by running in CLI:

```bash
    python todo_agent.py
```
 You should see:
 ```bash
    Agentic To-Do. say exit or quit to stop.
```