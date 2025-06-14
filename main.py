import os
import asyncio
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import AsyncOpenAI
from agents import (
    Agent, Runner, RunContextWrapper,
    handoff, function_tool, OpenAIChatCompletionsModel,
    set_tracing_disabled
)
from agents.extensions import handoff_filters

# -----------------------------
# Load environment and tracing
# -----------------------------
load_dotenv()
set_tracing_disabled(True)

# -----------------------------
# OpenAI / Gemini API Setup
# -----------------------------
Provider = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash-exp",
    openai_client=Provider,
)

# -----------------------------
# Pydantic Models for Inputs
# -----------------------------
class BillingInput(BaseModel):
    customer_id: str
    billing_issue: str

class RefundInput(BaseModel):
    order_id: str
    refund_reason: str

class TechSupportInput(BaseModel):
    issue: str

class GeneralInfoInput(BaseModel):  # New Pydantic model for general info
    topic: str

class HandoffParams(BaseModel):
    input_type: str
    input_filter: Optional[str] = None
    on_handoff: Optional[str] = None

# -----------------------------
# Tool Functions (function_tool)
# -----------------------------
@function_tool
def billing_status_tool(data: BillingInput) -> str:
    """
    Billing Confirmation Tool
    Reply: Your billing has been successfully processed.
    """
    return (
        f"Billing for Customer ID {data.customer_id} regarding '"
        f"{data.billing_issue}' has been cleared successfully."
    )

@function_tool
def refund_status_tool(data: RefundInput) -> str:
    """
    Refund Tool
    Reply: Your refund request has been completed successfully.
    """
    return (
        f"Refund for Order ID {data.order_id} has been processed successfully "
        f"due to: {data.refund_reason}."
    )

@function_tool
def general_info_tool(data: GeneralInfoInput) -> str:  # Updated to use Pydantic model
    """
    General Info Tool
    Replies with FAQ answers: shipping status, account details, etc.
    """
    return f"Here is the information you requested about '{data.topic}'."

@function_tool
def tech_support_tool(data: TechSupportInput) -> str:
    """
    Tech Support Tool
    Provides basic troubleshooting steps for common issues.
    """
    return f"Here are some steps to troubleshoot your issue with '{data.issue}'."

# -----------------------------
# Handoff Decision Logic
# -----------------------------
async def handoff_decision(params: HandoffParams) -> str:
    """
    Decide whether to escalate to a human agent.
    """
    sensitive = ["legal", "complaint", "technical"]
    if params.input_type in sensitive:
        return (
            "I am escalating your request to a human agent "
            "because it involves a sensitive or complex matter."
        )
    return "AI agent handled the query successfully."

# -----------------------------
# Agent Handlers
# -----------------------------
async def custom_billing_handler(
    ctx: RunContextWrapper[None],
    data: BillingInput
) -> str:
    """
    Custom handler for billing agent.
    """
    return f"Let me check your billing status... {billing_status_tool(data)}"

async def custom_refund_handler(
    ctx: RunContextWrapper[None],
    data: RefundInput
) -> str:
    """
    Custom handler for refund agent.
    """
    return f"Let me process your refund... {refund_status_tool(data)}"

async def custom_tech_support_handler(
    ctx: RunContextWrapper[None],
    data: TechSupportInput
) -> str:
    """
    Custom handler for tech support agent.
    """
    return f"Let me assist with your technical issue... {tech_support_tool(data)}"

async def custom_info_handler(
    ctx: RunContextWrapper[None],
    data: GeneralInfoInput  # Updated to use Pydantic model
) -> str:
    """
    Custom handler for info agent.
    """
    return f"Let me get that information for you... {general_info_tool(data)}"

# -----------------------------
# Agent Definitions
# -----------------------------
billing_agent = Agent(
    name="billing_agent",
    instructions=(
        "You are BillingAgent, a polite and professional customer support agent. "
        "Confirm billing issues, apologize for any inconvenience, and provide clear next steps."
    ),
    model=model,
)

refund_agent = Agent(
    name="refund_agent",
    instructions=(
        "You are RefundAgent, a courteous and empathetic support agent. "
        "Process refund requests, confirm details, and apologize for any inconvenience."
    ),
    model=model,
)

info_agent = Agent(
    name="info_agent",
    instructions=(
        "You are InfoAgent, a helpful FAQ agent. "
        "Provide concise and accurate answers to general inquiries about shipping, accounts, and policies."
    ),
    model=model,
)

tech_support_agent = Agent(
    name="tech_support_agent",
    instructions=(
        "You are TechSupportAgent, a knowledgeable and patient support agent. "
        "Assist with technical issues, provide troubleshooting steps, and escalate complex problems."
    ),
    model=model,
)

main_agent = Agent(
    name="main_agent",
    instructions=(
        "You are MainSupportAgent. Greet the user warmly, ask how you can assist, "
        "and use specialized agents/tools to handle requests about billing, refunds, general info, or technical support. "
        "If unsure, ask clarifying questions or escalate politely."
    ),
    model=model,
    handoffs=[
        handoff(
            agent=billing_agent,
            tool_name_override="billing_status_tool",
            tool_description_override="Handles customer billing inquiries.",
            input_type=BillingInput,
            input_filter=handoff_filters.remove_all_tools,
            on_handoff=custom_billing_handler,
        ),
        handoff(
            agent=refund_agent,
            tool_name_override="refund_status_tool",
            tool_description_override="Handles refund requests.",
            input_type=RefundInput,
            input_filter=handoff_filters.remove_all_tools,
            on_handoff=custom_refund_handler,
        ),
        handoff(
            agent=info_agent,
            tool_name_override="general_info_tool",
            tool_description_override="Provides general FAQs and information.",
            input_type=GeneralInfoInput,  # Updated to use Pydantic model
            on_handoff=custom_info_handler,
        ),

        handoff(
            agent=tech_support_agent,
            tool_name_override="tech_support_tool",
            tool_description_override="Handles technical support queries.",
            input_type=TechSupportInput,
            input_filter=handoff_filters.remove_all_tools,
            on_handoff=custom_tech_support_handler,
        ),
    ],
)

# -----------------------------
# Main Chat Loop
# -----------------------------
async def main():
    print("Welcome to ACME Corp Customer Support! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == 'exit':
            print("Thank you for contacting ACME Corp. Have a great day!")
            break

        response = await Runner.run(
            starting_agent=main_agent,
            input=user_input,
        )
        print(f"SupportBot: {response.final_output}\n")

if __name__ == "__main__":
    asyncio.run(main())