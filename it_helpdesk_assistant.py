#!/usr/bin/env python3
"""
Local AI‑Powered IT Helpdesk Assistant
=====================================

This script implements a lightweight, offline‑capable IT helpdesk assistant.  It
provides step‑by‑step troubleshooting suggestions using a locally installed
language model (via the `ollama` command), captures basic system context, and
logs each interaction for later review by an IT team.  If no local LLM is
available, the assistant still functions and returns a stub response to avoid
reaching out to any cloud services.

Features
--------

* Uses a local LLM through the `ollama` CLI to generate answers.  You can
  configure the model name via a command line flag (defaults to `phi`).
* Collects local system information (username, hostname, IP address) and
  timestamps the interaction.
* Accepts user queries via a simple command line interface.
* Logs all interactions to JSON files in a ``Tickets/Unprocessed`` folder,
  including user input, AI response, system information, severity and
  department tags.
* Assigns a simple severity level based on keywords found in the user’s
  question and attempts to infer a department from common terms.
* Designed to run entirely offline; no network calls are made by this
  script (unless `ollama` itself pulls model data).  The fallback response
  ensures functionality even without an installed model.

Usage
-----

Run the script directly from your terminal.  For example:

```
python3 it_helpdesk_assistant.py
```

The assistant will display a prompt for your helpdesk question.  Type your
issue, press Enter, and the assistant will respond.  The conversation is
logged automatically.  To exit, press Ctrl‑C or send an empty input.

Environment
-----------

This script does not require any external Python dependencies beyond the
standard library.  It relies on the presence of the ``ollama`` command in
your system’s PATH.  If `ollama` is not installed, the assistant falls back
to a predefined stub response instructing the user to contact IT staff.

"""

import json
import os
import re
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import getpass


def get_system_info() -> dict:
    """Collect basic system information.

    Returns a dictionary containing the logged‑in username, the computer
    hostname, the first non‑loopback IP address and the current timestamp.
    """
    username = getpass.getuser()
    hostname = socket.gethostname()
    ip_address = None

    try:
        # Attempt to find a non‑loopback IPv4 address.  This may return
        # 127.0.0.1 if the machine isn’t connected to a network.
        ip_address = socket.gethostbyname(hostname)
        if ip_address.startswith("127."):
            # Try probing through all interfaces to find a more meaningful IP.
            for addr_info in socket.getaddrinfo(hostname, None):
                family, _, _, _, sockaddr = addr_info
                if family == socket.AF_INET:
                    candidate = sockaddr[0]
                    if not candidate.startswith("127."):
                        ip_address = candidate
                        break
    except socket.gaierror:
        ip_address = "unknown"

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "username": username,
        "computer_name": hostname,
        "ip_address": ip_address or "unknown",
    }


def build_prompt(system_info: dict, user_input: str) -> str:
    """Construct the full prompt for the language model.

    Combines a system instruction that sets the role and behaviour of the
    assistant with the user’s question.  Includes a caution to avoid
    recommending any internet‑dependent solutions.
    """
    system_prompt = (
        "You are an IT helpdesk assistant (Tier 1 support agent). "
        "Provide clear, step‑by‑step troubleshooting instructions for "
        "common technical issues. Your answers must not rely on any cloud "
        "or internet‑based services. Never ask the user to go online. "
        "Be concise and polite, and always include a check‑in question at "
        "the end to see if the instructions helped."
    )
    combined_prompt = f"{system_prompt}\nUser: {user_input}\nAssistant:"
    return combined_prompt


def query_local_llm(prompt: str, model: str = "phi") -> str:
    """Send a prompt to a local language model via the `ollama` CLI.

    If the `ollama` command is unavailable or errors occur, a stub reply
    explaining the limitation is returned.  The `model` parameter specifies
    which model to use (e.g., 'phi', 'llama2', 'mistral').
    """
    try:
        # Run `ollama run <model>` with the prompt via stdin and capture the output.
        # We use text mode for easier string handling.
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            # Sometimes the LLM might echo the prompt or include tags; clean up.
            response = result.stdout.strip()
            # Remove prompt if echo appears in output (some LLM shells echo input).
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            return response or "I’m sorry, I don’t have a response at the moment."
        else:
            # Non‑zero return code, treat as failure.
            return (
                "I’m sorry, I couldn’t generate a response because the local "
                "model encountered an error. Please contact your IT staff for assistance."
            )
    except FileNotFoundError:
        # `ollama` not installed or not in PATH.
        return (
            "It appears the local language model is not installed. Please ask "
            "your IT administrator to install an Ollama‑compatible model. In the "
            "meantime, refer to internal troubleshooting guides or contact IT directly."
        )
    except Exception as exc:
        # Generic fallback for unexpected errors.
        return (
            "An unexpected error occurred while accessing the local model. "
            "Please contact IT support."
        )


def determine_severity(user_input: str) -> str:
    """Determine a simple severity level based on keywords in the user input."""
    normalized = user_input.lower()
    high_keywords = ["urgent", "immediately", "asap", "crash", "fatal", "severe"]
    medium_keywords = ["not working", "error", "problem", "issue", "won't", "cant", "can't"]
    for kw in high_keywords:
        if kw in normalized:
            return "high"
    for kw in medium_keywords:
        if kw in normalized:
            return "medium"
    return "low"


def infer_department(user_input: str) -> str:
    """Attempt to infer the user’s department from the query text."""
    normalized = user_input.lower()
    department_keywords = {
        "finance": ["invoice", "accounting", "finance", "budget", "expense"],
        "hr": ["payroll", "human resources", "benefits", "timesheet"],
        "marketing": ["campaign", "marketing", "advertising", "social media"],
        "it": ["server", "network", "vpn", "database", "email"],
        "sales": ["crm", "salesforce", "leads", "opportunity"],
        "operations": ["logistics", "operation", "inventory"]
    }
    for dept, keywords in department_keywords.items():
        for kw in keywords:
            if kw in normalized:
                return dept
    return "unknown"


def save_log(log_data: dict, log_dir: Path) -> Path:
    """Save a single interaction log to a JSON file in the given directory.

    Returns the path to the created file.  The filename includes the timestamp
    for uniqueness.  The directory is created if it does not exist.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    # Use a timestamp (YYYYMMDD_HHMMSS) for the filename.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ticket_{ts}.json"
    log_path = log_dir / filename
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)
    return log_path


def main():  # pragma: no cover
    """Main interactive loop for the helpdesk assistant."""
    print("Welcome to the Local IT Helpdesk Assistant. Type your question and press Enter."
          "\nPress Ctrl-C or enter an empty line to exit.\n")
    # Determine the location where tickets will be stored.
    base_dir = Path(__file__).resolve().parent
    ticket_dir = base_dir / "Tickets" / "Unprocessed"

    model = "phi"  # default model name; can be changed via CLI args
    if len(sys.argv) > 1:
        model = sys.argv[1]

    while True:
        try:
            user_input = input("Helpdesk> ")
        except KeyboardInterrupt:
            print("\nExiting. Goodbye!")
            break
        if not user_input.strip():
            print("No input received. Exiting. Goodbye!")
            break

        system_info = get_system_info()
        severity = determine_severity(user_input)
        department = infer_department(user_input)
        prompt = build_prompt(system_info, user_input)
        ai_response = query_local_llm(prompt, model=model)

        # Display the AI’s response.
        print(f"\nAI Response:\n{ai_response}\n")

        # Prepare log data.
        log_data = {
            **system_info,
            "user_input": user_input,
            "ai_response": ai_response,
            "severity": severity,
            "department": department,
        }
        log_path = save_log(log_data, ticket_dir)
        print(f"Session logged to: {log_path}\n")


if __name__ == "__main__":  # pragma: no cover
    main()