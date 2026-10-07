#!/usr/bin/env python3
"""Prototype generator for Hard-Tier Decision Archetypes and AlignBench Evaluator Archetypes."""

import random

from classone.data.dataset import bundle_multi_task_items
from classone.schemas import ChoiceQuestion, NoulQuestion


def generate_limitation_sample():
    start_year = random.randint(2021, 2023)
    start_month = random.randint(1, 12)
    start_day = random.randint(1, 28)
    filing_year = start_year + random.choice([2, 3, 4])
    filing_month = random.randint(1, 12)
    filing_day = random.randint(1, 28)

    # Calculate months difference
    months_diff = (filing_year - start_year) * 12 + (filing_month - start_month)
    is_time_barred = months_diff > 36

    state = {
        "rule": "Statute of Limitations Rule: Claims for property damage against contractors must be filed within 3 years (36 months) of the date of knowledge.",
        "intake": f"Client noticed damage on {start_year:04d}-{start_month:02d}-{start_day:02d}. Planned filing date is {filing_year:04d}-{filing_month:02d}-{filing_day:02d}.",
    }

    criteria = {
        "within_limitation": f"Filing is within the statutory 3-year limitation window ({months_diff} months <= 36 months).",
        "time_barred": f"Filing is time-barred as more than 3 years have elapsed ({months_diff} months > 36 months).",
        "insufficient_information": "The knowledge date cannot be determined from the intake records.",
    }

    target = "time_barred" if is_time_barred else "within_limitation"
    return bundle_multi_task_items(
        state=state,
        questions={
            "classification": ChoiceQuestion(
                instructions="How should this claim be classified under the statute of limitations rule?",
                criteria=criteria,
            ),
            "is_claim_barred": NoulQuestion(instructions="Is this claim time-barred by the statute of limitations?"),
        },
        targets={
            "classification": target,
            "is_claim_barred": 1.0 if is_time_barred else 0.0,
        },
    )


def generate_sublimit_sample():
    deductible = random.choice([500, 1000, 1500])
    sublimit = random.choice([10000, 15000, 20000])
    damage_amount = random.randint(18000, 35000)
    has_sublimit_endorsement = random.choice([True, False])

    state = {
        "policy": f"Homeowners Policy Form HM-3. Building deductible: ${deductible:,}. Endorsement W-1 imposes a ${sublimit:,} sublimit on concealed water seepage claims if attached.",
        "claim": f"Claim HX-991: Water seepage estimate is ${damage_amount:,}. Endorsement W-1 is {'attached and active' if has_sublimit_endorsement else 'not attached to this term'}.",
    }

    if has_sublimit_endorsement:
        target = f"pay_subject_to_{sublimit}_sublimit"
    else:
        target = "pay_full_estimate_less_deductible"

    criteria = {
        f"pay_subject_to_{sublimit}_sublimit": f"Loss is covered but payment is capped at the ${sublimit:,} endorsement sublimit.",
        "pay_full_estimate_less_deductible": f"Loss is covered without sublimit; full estimate of ${damage_amount:,} is paid less the ${deductible:,} deductible.",
        "deny_exclusion": "Loss is excluded entirely under general policy water exclusion.",
    }

    return bundle_multi_task_items(
        state=state,
        questions={
            "coverage_settlement": ChoiceQuestion(
                instructions="Determine how the building water-damage claim must be settled:", criteria=criteria
            ),
            "is_sublimit_applied": NoulQuestion(
                instructions="Does a policy endorsement sublimit cap the settlement payout?"
            ),
        },
        targets={
            "coverage_settlement": target,
            "is_sublimit_applied": 1.0 if has_sublimit_endorsement else 0.0,
        },
    )


def generate_teacher_judge_sample():
    is_correct = random.choice([True, False])
    question = "What is the capital of Australia?"
    gold = "Canberra"
    wrong_options = ["Sydney", "Melbourne", "Brisbane"]

    if is_correct:
        response = (
            "The capital of Australia is Canberra, which was chosen as a compromise between Sydney and Melbourne."
        )
    else:
        wrong_choice = random.choice(wrong_options)
        response = (
            f"I believe the capital of Australia is {wrong_choice}, as that is the largest and most prominent city."
        )

    state = {"question": question, "reference": gold, "response": response}

    return bundle_multi_task_items(
        state=state,
        questions={
            "accuracy_verdict": ChoiceQuestion(
                instructions="You are a teacher grading a student response against the true reference answer. Classify the response:",
                criteria={
                    "correct": "The response factually states the answer in the reference (regardless of extra context).",
                    "incorrect": "The response asserts a different answer that contradicts or conflicts with the reference.",
                },
            ),
            "is_factually_correct": NoulQuestion(
                instructions="Is the student answer factually correct according to the reference?"
            ),
        },
        targets={
            "accuracy_verdict": "correct" if is_correct else "incorrect",
            "is_factually_correct": 1.0 if is_correct else 0.0,
        },
    )


print("Generated sample limitation item:", generate_limitation_sample().targets)
print("Generated sample sublimit item:", generate_sublimit_sample().targets)
print("Generated sample teacher judge item:", generate_teacher_judge_sample().targets)
