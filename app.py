import streamlit as st
import uuid
import asyncio
import re
import io
import logging
from contextlib import redirect_stdout
from multi_agent_orchestrator.orchestrator import MultiAgentOrchestrator, OrchestratorConfig
from multi_agent_orchestrator.classifiers import ClassifierResult
from multi_agent_orchestrator.types import ConversationMessage
from multi_agent_orchestrator.agents import BedrockLLMAgent, BedrockLLMAgentOptions, AgentCallbacks, AgentResponse, SupervisorAgent, SupervisorAgentOptions
from job_search_tool import job_search_tool_description, job_search_tool_handler, job_search_tool_prompt


# === Custom log capture handler ===

def remove_ansi_escape_codes(text):
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)


class StringIOHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log_output = io.StringIO()
        self.setFormatter(logging.Formatter('%(message)s'))
        
    def emit(self, record):
        self.log_output.write(self.format(record) + '\n')
        
    def get_value(self):
        return self.log_output.getvalue()
        
    def clear(self):
        self.log_output = io.StringIO()

# Configure logging to capture multi_agent_orchestrator logs
log_capture_handler = StringIOHandler()
logger = logging.getLogger('multi_agent_orchestrator.utils.logger')
logger.addHandler(log_capture_handler)
logger.setLevel(logging.INFO)

# === Store agent callbacks separately ===
agent_callbacks = {}

# === Callback class for capturing agent thinking ===
class ThinkingTraceCallbacks(AgentCallbacks):
    def __init__(self, agent_name):
        self.agent_name = agent_name
        self.thinking_trace = []
        
    def on_llm_new_token(self, token: str) -> None:
        pass
        
    def record_thinking(self, thought: str):
        """Record a thinking step"""
        self.thinking_trace.append({"agent": self.agent_name, "thinking": thought})
    
    def get_thinking_trace(self):
        return self.thinking_trace

# === Function to parse agent communication from logs ===
def parse_agent_communication(log_output):
    thinking_trace = []
    
    # Pattern for supervisor sending messages
    sending_pattern = r"===>>>>> Supervisor sending ([^:]+): (.*)"
    log_output = remove_ansi_escape_codes(log_output)
    sending_matches = re.findall(sending_pattern, log_output)
    
    for agent, message in sending_matches:
        thinking_trace.append({
            "agent": "Supervisor Agent",
            "thinking": f"Sending request to {agent}: {message}"
        })
    
    # Pattern for supervisor receiving messages
    receiving_pattern = r"<<<<<===Supervisor received from ([^:]+):(.*?)(?=INFO:|$)"
    log_output = remove_ansi_escape_codes(log_output)
    receiving_matches = re.findall(receiving_pattern, log_output, re.DOTALL)
    
    for agent, message in receiving_matches:
        # Truncate long responses for the trace
        summary = message.strip()
        if len(summary) > 100:
            summary = summary[:100] + "..."
            
        thinking_trace.append({
            "agent": "Supervisor Agent",
            "thinking": f"Received response from {agent}: {summary}"
        })
        
    return thinking_trace


# ======================================================= Agents creation ===================================================================
def create_job_search_agent():
    callbacks = ThinkingTraceCallbacks("Job Search Agent")
    agent_callbacks["Job Search Agent"] = callbacks

    job_search_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
        name="Job Search Agent",
        streaming=False,
        description="Finds job listings in AI for contract work in your preferred location and timeframe.",
        tool_config={
            "tool": job_search_tool_description,
            "toolMaxRecursions": 5,
            "useToolHandler": job_search_tool_handler
        },
        callbacks=callbacks
    ))

    job_search_agent.set_system_prompt(job_search_tool_prompt)
    return job_search_agent

def create_house_broker_agent():
    callbacks = ThinkingTraceCallbacks("House Broker Agent")
    agent_callbacks["House Broker Agent"] = callbacks
    return BedrockLLMAgent(BedrockLLMAgentOptions(
        name="House Broker Agent",
        streaming=False,
        description="Recommends places to stay based on your preferences like budget and tranquility.",
        model_id="us.anthropic.claude-3-haiku-20240307-v1:0",
        callbacks=callbacks
    ))

def create_guide_agent():
    callbacks = ThinkingTraceCallbacks("Guide Agent")
    agent_callbacks["Guide Agent"] = callbacks
    return BedrockLLMAgent(BedrockLLMAgentOptions(
        name="Guide Agent",
        streaming=False,
        description="Recommends must-visit restaurants and activities in the area.",
        model_id="us.anthropic.claude-3-haiku-20240307-v1:0",
        callbacks=callbacks
    ))

def create_fashion_agent():
    callbacks = ThinkingTraceCallbacks("Fashion Agent")
    agent_callbacks["Fashion Agent"] = callbacks
    return BedrockLLMAgent(BedrockLLMAgentOptions(
        name="Fashion Agent",
        streaming=False,
        description="Suggests appropriate clothing based on the weather and season.",
        model_id="us.anthropic.claude-3-haiku-20240307-v1:0",
        callbacks=callbacks
    ))

# === Initialize Agents ===
job_agent = create_job_search_agent()
house_agent = create_house_broker_agent()
guide_agent = create_guide_agent()
fashion_agent = create_fashion_agent()

# === Create supervisor with custom prompt to expose thinking ===
supervisor_callbacks = ThinkingTraceCallbacks("Supervisor Agent")
agent_callbacks["Supervisor Agent"] = supervisor_callbacks

# Enhanced supervisor prompt to expose thinking
SUPERVISOR_PROMPT = """
You are a supervisor agent responsible for routing requests to the best-suited agent.
Before delegating to a specialist, explain your thinking process in detail.

The available agents are:
1. Job Search Agent: Finds job listings in AI for contract work in a preferred location and timeframe.
2. House Broker Agent: Recommends places to stay based on preferences like budget and tranquility.
3. Guide Agent: Recommends must-visit restaurants and activities in the area.
4. Fashion Agent: Suggests appropriate clothing based on the weather and season.

For each request, follow these steps:
1. THINKING: Analyze what the request is about
2. THINKING: Determine which agent would be most suitable
3. THINKING: Explain why you chose this agent
4. DELEGATION: Then select that agent

"""

lead_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
    name="SupervisorAgent",
    model_id="amazon.nova-pro-v1:0",
    description=SUPERVISOR_PROMPT,
    callbacks=supervisor_callbacks
))

supervisor = SupervisorAgent(SupervisorAgentOptions(
    name="Supervisor Agent",
    description="Routes user requests to the most relevant agent based on the topic and shows thinking process.",
    lead_agent=lead_agent,
    team=[job_agent, house_agent, guide_agent, fashion_agent],
    trace=True
))

# === Custom function to extract thinking from supervisor response ===
def extract_thinking_from_response(response_text):
    thinking_lines = []
    for line in response_text.split('\n'):
        if line.strip().startswith('THINKING:'):
            thinking_lines.append(line.strip().replace('THINKING:', '').strip())
    return thinking_lines

# ================================================================== Orchestrator ===================================================

orchestrator = MultiAgentOrchestrator(
    options=OrchestratorConfig(
        LOG_AGENT_CHAT=True,
        LOG_CLASSIFIER_CHAT=True,
        USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED=False
    )
)

# ==================================================================== Streamlit UI ==================================================

st.set_page_config(page_title="NomadBot 🌍", layout="wide")
st.title("🌍 Digital Nomad Companion")
st.caption("Helping you with work, stay, style, and play around the globe.")

# === Chat input and processing ===
user_input = st.chat_input("Ask about jobs, best places to stay, life abroad ✈️")


# === Chat session state ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.user_id = str(uuid.uuid4())

# === Display chat history ===
for message in st.session_state.chat_history:
    user_msg = message["user_input"]
    response = message["response"]
    thinking_trace = message.get("thinking_trace", [])

    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        agent_name = response.metadata.agent_name if hasattr(response, 'metadata') and hasattr(response.metadata, 'agent_name') else "Unknown Agent"
        
        # Show the response
        if isinstance(response.output, str):
            st.markdown(f"**🤖 Response from `{agent_name}`**\n\n{response.output}")
        elif isinstance(response.output, ConversationMessage):
            st.markdown(f"**🤖 Response from `{agent_name}`**\n\n{response.output.content[0].get('text')}")

        # Show agent thinking process
        if thinking_trace:
            with st.expander("🧠 Agent Thinking Process", expanded=True):
                for thought in thinking_trace:
                    st.markdown(f"**{thought['agent']}**: {thought['thinking']}")


if user_input:
    # Reset thinking traces for all agents
    for callback in agent_callbacks.values():
        callback.thinking_trace = []
    
    # Clear log capture
    log_capture_handler.clear()
    
    with st.spinner("Thinking..."):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        classifier_result = ClassifierResult(selected_agent=supervisor, confidence=1.0)

        # Before processing, record initial thinking
        supervisor_callbacks.record_thinking("Analyzing user query: " + user_input)
        
        try:
            # Process the request
            response = loop.run_until_complete(
                orchestrator.agent_process_request(
                    user_input,
                    st.session_state.user_id,
                    st.session_state.session_id,
                    classifier_result
                )
            )
            
            # Get log output
            log_output = log_capture_handler.get_value()
            
            # Extract thinking trace from log output
            log_thinking_trace = parse_agent_communication(log_output)
            
            # Combine thinking traces
            thinking_trace = supervisor_callbacks.get_thinking_trace()
            thinking_trace.extend(log_thinking_trace)
            
            # Extract thinking from response content
            if isinstance(response.output, ConversationMessage):
                content = response.output.content[0].get('text', '')
                thinking_lines = extract_thinking_from_response(content)
                for line in thinking_lines:
                    thinking_trace.append({
                        "agent": "Supervisor Agent", 
                        "thinking": line
                    })
            
            # Get specialized agent trace if we can identify which agent was used
            if hasattr(response, 'metadata') and hasattr(response.metadata, 'agent_name'):
                selected_agent = response.metadata.agent_name
                if selected_agent in agent_callbacks:
                    agent_trace = agent_callbacks[selected_agent].get_thinking_trace()
                    thinking_trace.extend(agent_trace)
                
                # Add final selection info
                supervisor_callbacks.record_thinking(f"Selected {selected_agent} to handle this request.")
        
            
        except Exception as e:
            # Fallback if any error happens
            response = type("FakeResponse", (), {})()
            response.output = "I'm sorry, I didn't understand exactly what you want. Could you try rephrasing it?"
            response.metadata = type("Metadata", (), {"agent_name": "Supervisor Agent"})()
            thinking_trace = supervisor_callbacks.get_thinking_trace()
            supervisor_callbacks.record_thinking("Could not process the input or agent failed.")

    # Save and rerun
    st.session_state.chat_history.append({
        "user_input": user_input,
        "response": response,
        "thinking_trace": thinking_trace
    })

    st.rerun()

# === Sidebar Info ===
with st.sidebar:
    st.header("🧠 Agents")
    st.markdown("- **Job Search Agent**: Helps you find AI contract gigs.")
    st.markdown("- **House Broker Agent**: Finds peaceful and affordable places to live.")
    st.markdown("- **Guide Agent**: Recommends local spots and favorites.")
    st.markdown("- **Fashion Agent**: Dresses you for the season.")
    st.markdown("- **Supervisor**: Routes your questions smartly.")
    
    st.divider()
    st.subheader("🔎 Agent Thinking")
    st.markdown("Expand the **'🧠 Agent Thinking Process'** section under each response to see how the supervisor decides which agent should handle your request.")
    
    st.divider()
    if st.button("🔁 New Chat"):
        st.session_state.chat_history = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
