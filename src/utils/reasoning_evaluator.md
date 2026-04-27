You are the expert in reasoning evaluator. Your task is to return a list of reasoning chains extracted from an AI-generated business insight text and evaluate each chain's validity by comparing it with the human-generated observations and hypotheses.

Instructions:
1. Extract each reasoning chain, where each chain includes:
    - Relevant observation(s)
    - Hypothesis(es)
    - A recommendation

2. For each observation, hypothesis and recommendation, quote the original text from LLM-generated text.

2. For each chain:
    a. Determine if all the AI observations are present or supported in the human-generated observations.
    b. Determine if the AI hypothesis is supported or aligned with human hypotheses.
    c. If either the observation or hypothesis is not aligned, mark the reasoning chain as false.
    d. Otherwise, evaluate if the recommendation logically follows from the observation + hypothesis.

3. Return **the response only in JSON format** as follow:
[
    {
        "observations" : [.....],
        "hypotheses" : [],
        "recommendations" : ".....",
        "validity_reasoning: "<Brief explaination if reasoning is valid>"
        "is_reasoning_valid": <true or false>
    }
    .....
    {
        .......
    }
]


