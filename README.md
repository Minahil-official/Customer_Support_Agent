
---

````markdown
# 🤖 ACME Corp Multi-Agent Customer Support System

A robust, AI-driven customer support solution powered by OpenAI Agents SDK and Gemini models. This system intelligently routes user queries to specialized agents (Billing, Refunds, Tech Support, and General Info) and handles complex cases with dynamic handoff strategies.

---

## 📦 Features

- 🎯 Specialized AI agents for different support roles
- 🔄 Intelligent handoff between agents using the `handoff()` function
- 🛠 Tool-calling enabled with `@function_tool` decorators
- 🔐 Environment variable support with `.env` loading
- 🤝 Context-aware decision-making for escalation
- 🧠 Gemini model integrated with OpenAI SDK interface
- 🚀 Asynchronous, interactive command-line chatbot experience

---

## 🛠 Technologies Used

- Python 3.10+
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
- Gemini API via `openai.AsyncOpenAI`
- Pydantic for data validation
- `dotenv` for environment management
- `asyncio` for asynchronous I/O

---

## ⚙️ Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/customer-support-agents.git
   cd customer-support-agents
````

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Create a `.env` file:**

   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

4. **Run the support chatbot:**

   ```bash
   python support_bot.py
   ```

---

## 🧩 How It Works

The system uses a primary `main_agent` to interact with users and route requests to four specialized sub-agents:

* `billing_agent`
* `refund_agent`
* `info_agent`
* `tech_support_agent`

Each agent uses a dedicated handler and tool registered with the `@function_tool` decorator. The `handoff()` method from the OpenAI Agents SDK routes context and control to the appropriate agent.

### 🔁 Example Handoff Definition

```python
handoff(
    agent=billing_agent,
    tool_name_override="billing_status_tool",
    tool_description_override="Handles customer billing inquiries.",
    input_type=BillingInput,
    input_filter=handoff_filters.remove_all_tools,
    on_handoff=custom_billing_handler,
)
```

---

## ✨ Code Usage Example

### Agent Initialization

```python
main_agent = Agent(
    name="main_agent",
    instructions="You are MainSupportAgent. Greet the user warmly...",
    model=model,
    handoffs=[...],
)
```

### Tool Function

```python
@function_tool
def billing_status_tool(data: BillingInput) -> str:
    return f"Billing for Customer ID {data.customer_id} ... has been cleared successfully."
```

### Main Loop

```python
while True:
    user_input = input("You: ")
    if user_input.strip().lower() == 'exit':
        break
    response = await Runner.run(
        starting_agent=main_agent,
        input=user_input,
    )
    print(f"SupportBot: {response.final_output}")
```

---

---

## 📄 License

This project is licensed under the MIT License.


