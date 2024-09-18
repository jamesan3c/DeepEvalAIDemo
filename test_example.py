from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualPrecisionMetric

passingThreshold = 0.7
input = "What is the return policy for these shoes?"
expectedOutput = "You can return the shoes within 30 days for a full refund."
answer_relevancy_metric = AnswerRelevancyMetric(threshold=passingThreshold, async_mode=False)
faithfulness_metric = FaithfulnessMetric(threshold=passingThreshold, async_mode=False)
context_precision_metric = ContextualPrecisionMetric(threshold=passingThreshold, async_mode=False)

def test_good_response():

    test_case = LLMTestCase(
        input=input,
        actual_output="We offer a 30-day full refund for all shoes purchased.",
        expected_output=expectedOutput,
        retrieval_context=["Our return policy allows for a 30-day full refund on all shoes purchased."]
    )

    assert_test(test_case, [answer_relevancy_metric, faithfulness_metric, context_precision_metric])

def test_misleading_response():
    
    test_case = LLMTestCase(
        input=input,
        actual_output="We offer a 60-day full refund for shoes.",
        expected_output=expectedOutput,
        retrieval_context=["Our return policy allows for a 30-day full refund on all shoes purchased."]
    )

    assert_test(test_case, [answer_relevancy_metric, faithfulness_metric, context_precision_metric])

def test_incorrect_response():
    
    test_case = LLMTestCase(
        input=input,
        actual_output="We do not offer returns for any purchased items.",
        expected_output=expectedOutput,
        retrieval_context=["Our return policy allows for a 30-day full refund on all shoes purchased."]
    )

    assert_test(test_case, [answer_relevancy_metric, faithfulness_metric, context_precision_metric])
