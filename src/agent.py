"""
LangChain agent setup for BiteBot restaurant assistant.

Memory architecture
-------------------
* MemorySaver (checkpointer) owns the full per-session conversation.
  We only hand agent.invoke() the NEW message; the checkpointer supplies
  the rest via thread_id.
* _TrimmedModel sits between the agent and OpenAI.  It keeps the system
  message + the last KEEP messages before every LLM call.  Without it the
  full thread history would be sent to OpenAI on every turn, which (a)
  burns tokens and (b) causes New Relic AI monitoring to label every
  request with the very first user message in the thread.

"""

import os
import re as _re
import logging
import warnings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, trim_messages


from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

from src.tools import all_tools, set_tool_context, get_tool_context, set_active_session

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Checkpointer – one instance, shared across the whole app.
# Each Flask session gets a unique thread_id so conversations are isolated.
# NOTE: MemorySaver is in-process only.  App restart clears all threads.
# For production persistence swap in SqliteSaver or a Redis checkpointer.
_checkpointer = MemorySaver()

# Agent factory
def create_bitebot_agent():
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OpenAI API key not found.")

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    system_prompt = """You are BiteBot, a friendly and conversational restaurant assistant.

        PERSONALITY:
        - Warm, enthusiastic, and helpful
        - Present information naturally in paragraphs, not bullet lists

        TOOLS:
        - search_restaurants_tool: Find restaurants
        - get_restaurant_details_tool: Get full info
        - check_availability_tool: Check hours OR table availability
        - make_reservation_tool: Book tables

        CRITICAL RULE #1: PASS DATES EXACTLY AS USER SAYS THEM

        CORRECT: date="today", date="tomorrow", date="this friday", date="next thursday"
        WRONG:   date="2026-02-03"  — never calculate dates yourself, the tool handles it

        CRITICAL RULE #2: RESERVATION WORKFLOW

        1. Call check_availability_tool first (pass date exactly as user said)
        2. Present availability to user
        3. Ask: "Would you like me to book this table?"
        4. Wait for confirmation ("yes", "sure", "book it", etc.)
        5. Ask: "What name should I put the reservation under?"
        6. Wait for their real name
        7. ONLY THEN call make_reservation_tool (just pass name + customer details, date/time are automatic)

        NEVER use placeholder names like "Guest" or "User" — the tool will reject them.

        Be conversational and helpful!"""

    logger.info("Creating agent with 4 tools")
    for t in all_tools:
        logger.info(f"  - {t.name}")

    agent = create_agent(
        model,
        all_tools,
        system_prompt=system_prompt,
        checkpointer=_checkpointer,
    )
    return agent

# Run
def run_agent(agent, user_message: str, thread_id: str, tool_context: dict = None) -> dict:
    """Invoke the agent with a single new message.

    The checkpointer owns conversation history -- we do NOT replay it.

    Args:
        agent:        The compiled agent graph.
        user_message: What the user typed this turn.
        thread_id:    One per Flask session; scopes the agent's memory.
        tool_context: Inter-tool state (e.g. availability) persisted in
                      Flask session separately from the agent's memory.

    Returns:
        output           -- the assistant's reply (clean text).
        tool_context     -- updated inter-tool state to persist in session.
        reservation_json -- raw JSON string if a reservation was confirmed, else None.
    """
    # Restore tool context from previous request
    if tool_context:
        for key, value in tool_context.items():
            if value is not None:
                set_tool_context(key, value)

    logger.info(f"[AGENT] thread={thread_id} | message={user_message}")

    # Bind this execution context to the session's thread_id.  Must happen
    # before invoke() — ContextVar propagates into the worker threads that
    # langgraph uses to run tools, so they'll land in the right bucket.
    set_active_session(thread_id)

    config = {"configurable": {"thread_id": thread_id}}
    response = agent.invoke(
        {"messages": [HumanMessage(content=user_message)]},
        config=config,
    )
    response_messages = response.get("messages", [])

    logger.info(f"[AGENT] {len(response_messages)} messages in thread after invoke")

    # Debug log -- helpful for tracing tool-call loops
    for i, msg in enumerate(response_messages):
        msg_type = type(msg).__name__
        logger.info(f"[AGENT] [{i}] {msg_type} | {str(getattr(msg, 'content', ''))[:120]}")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            logger.info(f"[AGENT]     tool_calls: {msg.tool_calls}")

    # Final output = last non-empty AIMessage
    output = ""
    for msg in reversed(response_messages):
        if type(msg).__name__ == "AIMessage" and msg.content:
            output = msg.content
            break

    # Snapshot tool context for session persistence
    updated_tool_context = {"availability": get_tool_context("availability")}

    # Pull reservation JSON out of ToolMessages (the IMPORTANT line lives
    # there, not in the LLM's reply -- see tools.py make_reservation_tool).
    reservation_json = None
    for msg in response_messages:
        if type(msg).__name__ == "ToolMessage":
            match = _re.search(
                r"IMPORTANT: This reservation data includes: ({.*})",
                str(msg.content),
            )
            if match:
                reservation_json = match.group(1)

    logger.info(f"[AGENT] output={output[:200]}")
    return {
        "output": output,
        "tool_context": updated_tool_context,
        "reservation_json": reservation_json,
    }
