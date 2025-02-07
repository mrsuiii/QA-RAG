from retrivial import Retrivial
from langchain_ollama import OllamaLLM  # or use your preferred import if needed

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response?
"""

def query_and_validate(question: str, expected_response: str) -> bool:
    """
    Executes the query using Retrivial, evaluates the answer with the OllamaLLM model,
    and returns True if the evaluation returns 'true', False if 'false'.
    """
    # Generate the actual response using your retrieval agent
    agent = Retrivial(question)
    response_text = agent.run()

    # Prepare the evaluation prompt
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response,
        actual_response=response_text
    )
    
    # Instantiate the OllamaLLM client with the correct base URL and settings
    model = OllamaLLM(
        base_url="http://ollama:11434/",  # Ensure trailing slash if required
        model="llama3.2:1b",
        temperature=0.1,
    )
    
    # Get the evaluation result from the model
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print("\n=== Evaluation Prompt ===")
    print(prompt)
    print("=== Evaluation Result ===")
    print(evaluation_results_str_cleaned)

    # Determine if the response is correct or not based on the evaluation result
    if "true" in evaluation_results_str_cleaned:
        print("\033[92m" + "Result: CORRECT" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        print("\033[91m" + "Result: INCORRECT" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false': {evaluation_results_str_cleaned}"
        )

def run_tests():
    # List your test cases in a structured way
    test_cases = [
        {
            "name": "Monopoly Rules",
            "question": "How much total money does a player start with in Monopoly? (Answer with the number only)",
            "expected_response": "$1500",
        },
        {
            "name": "Ticket to Ride Rules",
            "question": "How many points does the longest continuous train get in Ticket to Ride? (Answer with the number only)",
            "expected_response": "10 points",
        }
    ]

    correct_count = 0
    total_tests = len(test_cases)

    # Run each test case and track successes
    for test in test_cases:
        print(f"\n--- Running Test: {test['name']} ---")
        try:
            result = query_and_validate(test["question"], test["expected_response"])
            if result:
                correct_count += 1
        except Exception as e:
            print("\033[91m" + f"Test {test['name']} failed with exception: {e}" + "\033[0m")
    
    # Calculate and print the accuracy
    accuracy = (correct_count / total_tests) * 100 if total_tests > 0 else 0
    print("\n=== Test Summary ===")
    print(f"Passed {correct_count} out of {total_tests} tests. Accuracy: {accuracy:.2f}%")
