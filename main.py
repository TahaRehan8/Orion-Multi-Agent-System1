from agents.finance_agent import ask_finance
from agents.hr_agent import ask_hr
from agents.scheduler_agent import ask_scheduler
from backend.orchestrator import coordinate

def print_separator():
    print("\n" + "=" * 60 + "\n")

def test_individual_agent():
    print("Multi-Agent RAG System")
    print("Agents: scheduler, hr, finance")
    print("Type 'quit' to exit, 'switch <agent>' to change agent")
    print_separator()

    current_agent = "scheduler"

    while True:
        query = input(f"[{current_agent}] Enter query: ").strip()

        if query.lower() == 'quit':
            break

        if query.lower().startswith('switch '):
            agent = query.split(' ', 1)[1].lower()
            if agent in ['scheduler', 'hr', 'finance']:
                current_agent = agent
                print(f"Switched to {agent} agent.")
            else:
                print("Invalid agent. Choose: scheduler, hr, finance")
            continue

        if not query:
            continue

        try:
            if current_agent == 'scheduler':
                response = ask_scheduler(query)
            elif current_agent == 'hr':
                response = ask_hr(query)
            elif current_agent == 'finance':
                response = ask_finance(query)

            print(f"\nResponse:\n{response}")
        except Exception as e:
            print(f"Error: {e}")

        print_separator()

def test_orchestrator_agent():
    print("Multi-Agent RAG System: Custom Orchestrator Mode")
    print("Type 'quit' to exit, or 'mode all'/'mode average'/'mode best' to switch orchestration mode.")
    print_separator()

    mode = "average"
    while True:
        query = input(f"[orchestrator | mode={mode}] Enter query: ").strip()
        if query.lower() == 'quit':
            break
        if query.lower().startswith('mode '):
            requested_mode = query.split(" ", 1)[1].strip()
            if requested_mode in ("all", "best", "average"):
                mode = requested_mode
                print(f"Switched to orchestrator mode: {mode}")
            else:
                print("Invalid mode. Choose 'all', 'average' or 'best'.")
            continue
        if not query:
            continue
        try:
            response = coordinate(query)
            response = response.final_response if hasattr(response, "final_response") else response
            print(f"\nResponse:\n{response}")
        except Exception as e:
            print(f"Error: {e}")
        print_separator()

if __name__ == "__main__":
    print("Choose Orchestrator:")
    print("1: Custom Mistral Orchestrator")
    print("2: Test Individual Agent")
    choice = input("Enter choice (1-2): ").strip()
    
    if choice == "1":
        test_orchestrator_agent()
    else:
        test_individual_agent()
