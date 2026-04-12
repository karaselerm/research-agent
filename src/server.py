import argparse
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from executor import Executor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the A2A purple agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    args = parser.parse_args()

    skill = AgentSkill(
        id="research_ml_reasoning",
        name="Research & ML Reasoning",
        description="Answers research-style and ML-engineering prompts with concise structured reasoning.",
        tags=["research", "analysis", "ml", "reasoning"],
        examples=[
            "Summarize the likely causes of underfitting and suggest fixes.",
            "Compare two machine learning approaches for a practical task.",
            "Give a concise research-style answer with assumptions and next steps.",
        ],
    )

    agent_card = AgentCard(
        name="Kartoshechko Research Agent",
        description="A purple agent for the AgentBeats Research track, optimized for concise structured analytical responses.",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()