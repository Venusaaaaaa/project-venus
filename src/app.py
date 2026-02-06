from agent import create_agent

if __name__ == "__main__":
    agent = create_agent()

    print("\n=== Car Sensor Troubleshooting Assistant ===\n")
    user_input = input("Enter your sensor problem: ")

    result = agent.invoke({"input": user_input})

    print("\n=== ANSWER ===\n")
    print(result["final_answer"])
