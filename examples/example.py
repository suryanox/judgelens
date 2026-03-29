from judgelens import EvalSample
from judgelens.auditor import JudgeAuditor

samples = [
    EvalSample(
        sample_id="q1",
        input="What is the capital of France?",
        response="The capital of France is Paris, a city known for the Eiffel Tower.",
        ground_truth="Paris",
        response_model="gpt-4o",
    ),
    EvalSample(
        sample_id="q2",
        input="Explain what a neural network is in one sentence.",
        response="A neural network is a computational system loosely inspired by the human brain, consisting of layers of interconnected nodes that learn patterns from data.",
        ground_truth="A neural network is a machine learning model made of layers of interconnected nodes that learn to recognize patterns in data.",
        response_model="gpt-3.5-turbo",
    ),
    EvalSample(
        sample_id="q3",
        input="What is 12 multiplied by 8?",
        response="12 multiplied by 8 equals 96.",
        ground_truth="96",
        response_model="gpt-4o",
    ),
    EvalSample(
        sample_id="q4",
        input="Who wrote the play Hamlet?",
        response="Hamlet was written by the English playwright William Shakespeare, likely around 1600.",
        ground_truth="William Shakespeare",
        response_model="claude-opus-4-6",
    ),
    EvalSample(
        sample_id="q5",
        input="What does HTTP stand for?",
        response="HTTP stands for HyperText Transfer Protocol, the foundation of data communication on the World Wide Web.",
        ground_truth="HyperText Transfer Protocol",
        response_model="gpt-4o",
    ),
]

PROMPT="""\
You are an expert evaluator. Given a question, a model's response, \
and a reference answer, score the response on a scale of 0 to 10.

Question:
{input}

Model Response:
{response}

Reference Answer:
{ground_truth}

Evaluate for correctness, completeness, and clarity relative to the reference.

Respond in EXACTLY this JSON format with no other text:
{{"score": <integer 0-10>, "reasoning": "<one sentence>"}}"""


auditor = JudgeAuditor(
    judge_model="gpt-4o",
    judge_prompt=PROMPT,
)

print("Running audit... (this makes several API calls per probe)\n")
report = auditor.run(samples)

print(report.summary())
print()

print(f"trust_score          : {report.trust_score}")
print(f"grade                : {report.grade}")
print(f"position_bias        : {report.position_bias}")
print(f"verbosity_bias       : {report.verbosity_bias}")
print(f"self_preference_bias : {report.self_preference_bias}")
print(f"calibration_score    : {report.calibration_score}")
print()

print("Per-sample raw scores:")
for s in report.per_sample:
    print(f"  [{s.sample_id}] raw={s.raw_score:.2f}  corrected={s.corrected_score:.2f}  reasoning={s.judge_reasoning[:60]}...")

print("\nProbe detail (position bias):")
for p in report.probe_results:
    if p.probe_name == "position_bias" and not p.skipped:
        print(f"  flip_rate   : {p.detail['flip_rate']:.2%}")
        print(f"  mean_delta  : {p.detail['mean_score_delta']:.3f}")
        print(f"  n_flipped   : {p.detail['n_flipped']} / {p.detail['n_tested']}")
