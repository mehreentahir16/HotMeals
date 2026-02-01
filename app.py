"""
BiteBot Restaurant Agent - Streamlit UI

A conversational interface for the restaurant recommendation agent.
"""

import streamlit as st
from src.agent import create_bitebot_agent, run_agent

# Page configuration
st.set_page_config(
    page_title="BiteBot Restaurant Agent",
    page_icon="🍽️",
    layout="wide"
)

# Title and description
st.title("🍽️ BiteBot Restaurant Agent")
st.markdown("""
Welcome to BiteBot! I'm your AI restaurant assistant powered by real Yelp data.

Ask me to:
- 🔍 Find restaurants by cuisine, location, or price
- ℹ️ Get detailed information about specific restaurants
- ⏰ Check if restaurants are currently open
- 💡 Get personalized recommendations

**Example queries:**
- "Find me Italian restaurants in Philadelphia"
- "Show me affordable Chinese food in Tampa"  
- "Is Vetri Cucina open right now?"
- "Tell me more about Han Dynasty"
""")

st.divider()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    with st.spinner("Initializing BiteBot agent..."):
        try:
            st.session_state.agent = create_bitebot_agent()
            st.success("✅ Agent initialized successfully!")
        except FileNotFoundError as e:
            st.error(f"❌ Database not found: {e}")
            st.info("Please run: `python scripts/prepare_data.py` to create the database.")
            st.stop()
        except ValueError as e:
            st.error(f"❌ Configuration error: {e}")
            st.info("Please create a `.env` file with your `OPENAI_API_KEY`")
            st.stop()
        except Exception as e:
            st.error(f"❌ Error initializing agent: {e}")
            st.stop()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show tool calls if available
        if message.get("show_tools") and message.get("intermediate_steps"):
            with st.expander("🔧 View Agent's Tool Calls", expanded=False):
                for step in message["intermediate_steps"]:
                    action, observation = step
                    st.markdown(f"**Tool:** `{action.tool}`")
                    st.code(action.tool_input, language="json")
                    st.markdown(f"**Result:**")
                    st.text(observation[:500] + ("..." if len(observation) > 500 else ""))
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask me about restaurants..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get agent response with full conversation history
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = run_agent(
                st.session_state.agent, 
                prompt,
                conversation_history=st.session_state.messages  # Pass existing messages
            )
            
            # Extract output and intermediate steps
            output = response.get("output", "I apologize, but I couldn't process that request.")
            intermediate_steps = response.get("intermediate_steps", [])
            
            # Display response
            st.markdown(output)
            
            # Show tool calls in expander
            if intermediate_steps:
                with st.expander("🔧 View Agent's Tool Calls", expanded=False):
                    for step in intermediate_steps:
                        action, observation = step
                        st.markdown(f"**Tool:** `{action.tool}`")
                        st.code(action.tool_input, language="json")
                        st.markdown(f"**Result:**")
                        st.text(observation[:500] + ("..." if len(observation) > 500 else ""))
                        st.divider()
    
    # Add assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": output,
        "show_tools": True,
        "intermediate_steps": intermediate_steps
    })

# Sidebar
with st.sidebar:
    st.header("About BiteBot")
    st.markdown("""
    **BiteBot** is an agentic AI system built for:
    - 📚 "The Software Engineer's Guide to Agentic AI Systems & Observability"
    
    **Architecture:**
    - 🧠 LLM: OpenAI GPT-4o-mini
    - 🔧 Framework: LangChain
    - 🗄️ Database: SQLite (Yelp data)
    - 💬 UI: Streamlit
    
    **Tools:**
    - `search_restaurants_tool` - Find restaurants
    - `get_restaurant_details_tool` - Get info
    - `check_if_open_tool` - Check hours
    """)
    
    st.divider()
    
    if st.button("🔄 Reset Conversation"):
        st.session_state.messages = []
        st.session_state.agent = create_bitebot_agent()
        st.rerun()
    
    st.divider()
    
    st.markdown("**Dataset Statistics:**")
    try:
        from src.database import get_connection
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM restaurants")
        total = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(DISTINCT city) FROM restaurants")
        cities = cursor.fetchone()[0]
        cursor.execute("SELECT AVG(stars) FROM restaurants")
        avg_rating = cursor.fetchone()[0]
        conn.close()
        
        st.metric("Total Restaurants", f"{total:,}")
        st.metric("Cities", f"{cities:,}")
        st.metric("Avg Rating", f"{avg_rating:.2f}⭐")
    except:
        st.info("Database stats unavailable")
    
    st.divider()
    st.caption("💡 Tip: Be specific about location and preferences for better recommendations!")