from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval.test_case import LLMTestCase
from deepeval import assert_test

correctness_metric = GEval(
    name="Correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    evaluation_steps=[
        "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
        "Vague language, or contradicting OPINIONS, are OK"
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
)

nonvague_correctness_metric = GEval(
    name="Non-vagueness",
    criteria="Determine whether the actual output is vague and unclear based on the expected output",
    evaluation_steps=[
        "You should also heavily penalize omission of detail",
        "Vague language, or contradicting OPINIONS, are NOT OK"
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
)

input = "The dog chased the cat up the tree, who ran up the tree?"
expected_output = "The cat."

def test_correct_response():
    test_case = LLMTestCase(
        input=input,
        actual_output="The cat ran up the tree.",
        expected_output=expected_output
    )
    assert_test(test_case, [correctness_metric, nonvague_correctness_metric])

def test_vague_response():
    test_case = LLMTestCase(
        input=input,
        actual_output="It depends, some might consider the cat, while others might argue the dog. But ultimately the cat is the answer",
        expected_output=expected_output
    )
    assert_test(test_case, [correctness_metric, nonvague_correctness_metric])