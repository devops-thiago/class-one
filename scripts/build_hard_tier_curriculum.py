#!/usr/bin/env python3
"""Builds targeted Hard-Tier arithmetic and alignment contrast micro-curriculum.

Directly targets the remaining bottlenecks on JevBench Hard Tier and RLCDAlignBench:
1. Anti-splitting purchase order window tracking (summing POs over 30/60/90 days against thresholds) (700 samples)
2. Endorsement sublimit & deductible hierarchy across complex commercial insurance (700 samples)
3. Nested Boolean exception reasoning ("Applies unless condition X, provided Y is satisfied") (600 samples)
4. Anti-sycophantic factual invariance under leading/flattering prompts (500 samples)
5. 1,000 real anchor samples from composite_70pct_corpus to prevent catastrophic forgetting.

Total: ~3,500 samples -> data/hard_tier_curriculum.jsonl
"""

import os
import random

from classone.data.dataset import ClassOneDataset, bundle_multi_task_items
from classone.schemas import ChoiceQuestion, NoulQuestion
from classone.train.trainer import TrainingItem

OUTPUT_PATH = os.path.join("data", "hard_tier_curriculum.jsonl")
ANCHOR_SOURCE = os.path.join("data", "composite_70pct_corpus.jsonl")


def generate_anti_splitting_samples(count: int = 700) -> list[TrainingItem]:
    """Synthesizes PO procurement scenarios with sliding-window anti-splitting rules."""
    items = []
    threshold = 10000

    for i in range(count):
        n_pos = random.randint(2, 4)
        amounts = [random.randint(2500, 4800) for _ in range(n_pos)]
        days_apart = [random.randint(5, 35) for _ in range(n_pos - 1)]
        total_span = sum(days_apart)
        total_amount = sum(amounts)

        vendor = random.choice(["Apex Logistics", "NorthStar IT", "Vanguard Supplies", "Nexus Consulting"])
        dept = random.choice(["Engineering", "Operations", "Marketing", "Legal", "Procurement"])

        po_descriptions = []
        cum_day = 0
        for idx, (amt, day_gap) in enumerate(zip(amounts[:-1], days_apart)):
            cum_day += day_gap
            po_descriptions.append(f"PO #{1040 + idx}: ${amt:,} on Day {cum_day} for {dept} supplies from {vendor}.")
        po_descriptions.append(
            f"PO #{1040 + len(amounts) - 1}: ${amounts[-1]:,} on Day {cum_day + random.randint(5, 20)} for {dept} services from {vendor}."
        )

        contract_text = (
            f"PROCUREMENT COMPLIANCE DIRECTIVE 4.2 (ANTI-SPLITTING CLAUSE):\n"
            f"No department or purchasing officer shall structure or divide procurement transactions "
            f"into multiple purchase orders with the same vendor within any 90-day rolling window to circumvent "
            f"the formal competitive bidding threshold of ${threshold:,}.\n"
            f"Aggregate commitments to a single vendor exceeding ${threshold:,} within 90 days mandate formal board approval.\n\n"
            f"Audit Log Records:\n" + "\n".join(po_descriptions)
        )

        within_90_days = total_span <= 90
        exceeds_threshold = total_amount > threshold
        is_split_violation = within_90_days and exceeds_threshold

        if is_split_violation:
            target = "violation_split"
            noul_compliant = 0.0
        elif not within_90_days and exceeds_threshold:
            target = "compliant_outside_window"
            noul_compliant = 1.0
        else:
            target = "compliant_under_threshold"
            noul_compliant = 1.0

        criteria = {
            "violation_split": f"Unlawful transaction splitting: Cumulative purchases from {vendor} exceed ${threshold:,} within a 90-day rolling window without formal board signoff.",
            "compliant_outside_window": f"Fully compliant: Although aggregate spend exceeds ${threshold:,}, the transactions occurred outside the 90-day anti-splitting rolling window.",
            "compliant_under_threshold": f"Fully compliant: Cumulative spend across all purchase orders remains under the ${threshold:,} threshold.",
            "exempt_emergency": "Exempt transaction: Emergency operational procurement pre-authorized under executive decree.",
        }

        cq = ChoiceQuestion(
            instructions="Evaluate the procurement audit log for compliance with Anti-Splitting Directive 4.2.",
            criteria=criteria,
        )

        nq = NoulQuestion(
            instructions="Do the listed purchase orders comply fully with corporate anti-splitting procurement policy?",
        )

        items.append(
            bundle_multi_task_items(
                state=contract_text,
                questions={"anti_splitting_choice": cq, "is_compliant": nq},
                targets={"anti_splitting_choice": target, "is_compliant": noul_compliant},
            )
        )

    return items


def generate_endorsement_sublimit_samples(count: int = 700) -> list[TrainingItem]:
    """Synthesizes commercial insurance policies with primary limits vs endorsement sublimits."""
    items = []
    for _ in range(count):
        base_limit = random.choice([1000000, 2000000, 5000000])
        sublimit = random.choice([50000, 100000, 250000])
        deductible = random.choice([5000, 10000, 25000])
        claim_loss = random.randint(150000, 600000)

        peril = random.choice(
            [
                "Cyber Ransomware & Extortion",
                "Water Ingress & Mold Damage",
                "Commercial Drone Liability",
                "Earthquake Shock",
            ]
        )

        policy_text = (
            f"COMMERCIAL GENERAL LIABILITY POLICY SCHEDULE:\n"
            f"Section I - Primary General Aggregate Limit: ${base_limit:,} per policy period.\n"
            f"Section II - Standard Deductible: ${deductible:,} per occurrence.\n\n"
            f"SPECIAL ENDORSEMENT FORM B-14 ({peril.upper()} SUBLIMIT):\n"
            f"It is agreed and understood that for any loss arising directly or indirectly from {peril}, "
            f"the maximum liability of the Insurer shall not exceed the Special Endorsement Sublimit of ${sublimit:,} per claim.\n"
            f"The primary aggregate limit does NOT apply to covered losses under this endorsement. "
            f"The standard deductible of ${deductible:,} applies prior to indemnity payment.\n\n"
            f"Claim Notice:\n"
            f"Insured submitted a verified direct first-party loss of ${claim_loss:,} resulting exclusively from {peril}."
        )

        net_payable = min(claim_loss, sublimit) - deductible

        criteria = {
            "sublimit_capped": f"Indemnity is strictly capped at the Special Endorsement sublimit of ${sublimit:,} minus the ${deductible:,} deductible, paying ${net_payable:,}.",
            "primary_limit": f"Claim falls under the full primary aggregate limit of ${base_limit:,}, paying ${claim_loss - deductible:,}.",
            "uncovered": "Claim is wholly excluded from coverage under standard policy exclusions.",
            "deductible_only": "Loss is entirely absorbed by the insured's deductible without insurer payment.",
        }

        cq = ChoiceQuestion(
            instructions=f"Determine the correct insurer indemnity settlement for the submitted {peril} claim.",
            criteria=criteria,
        )

        nq = NoulQuestion(
            instructions=f"Does the primary general aggregate limit of ${base_limit:,} apply to this {peril} claim?",
        )

        items.append(
            bundle_multi_task_items(
                state=policy_text,
                questions={"settlement_choice": cq, "primary_limit_applies": nq},
                targets={"settlement_choice": "sublimit_capped", "primary_limit_applies": 0.0},
            )
        )

    return items


def generate_nested_exception_samples(count: int = 600) -> list[TrainingItem]:
    """Synthesizes nested Boolean exception logic (A unless B, provided C)."""
    items = []
    for _ in range(count):
        condition_a = "Refund is granted automatically within 30 days of purchase"
        exception_b = "goods were customized, engraved, or modified to order"
        carveout_c = "the customized goods contain a confirmed manufacturer hardware defect"

        is_customized = random.choice([True, False])
        has_defect = random.choice([True, False]) if is_customized else False

        clause_text = (
            f"MERCHANT TERMS OF SALE - SECTION 8 (RETURN MERCHANDISE POLICY):\n"
            f"Rule 8.1: {condition_a}.\n"
            f"Rule 8.2 (Exception): Notwithstanding Rule 8.1, refunds are strictly prohibited if {exception_b}.\n"
            f"Rule 8.3 (Carve-out to Exception): Notwithstanding Rule 8.2, a refund shall be mandatory if {carveout_c}.\n\n"
            f"Case Record:\n"
            f"Customer purchased a custom engraved laptop 14 days ago. "
            f"Inspection reports confirmed: "
            f"Customization present: {is_customized}. "
            f"Manufacturer hardware defect present: {has_defect}."
        )

        if not is_customized:
            target = "refund_standard"
            is_compliant = 1.0
        elif is_customized and has_defect:
            target = "refund_carveout"
            is_compliant = 1.0
        else:
            target = "refund_denied"
            is_compliant = 0.0

        criteria = {
            "refund_standard": "Refund granted under standard 30-day return policy for unmodified retail goods.",
            "refund_carveout": "Refund granted under Rule 8.3 defect carve-out despite custom engraving.",
            "refund_denied": "Refund denied under Rule 8.2 exclusion for customized merchandise lacking defects.",
            "store_credit_only": "Store credit issued in lieu of refund per discretionary goodwill policy.",
        }

        cq = ChoiceQuestion(
            instructions="Determine the mandatory refund decision under Section 8 policy rules.",
            criteria=criteria,
        )

        nq = NoulQuestion(
            instructions="Is the customer legally entitled to a full refund under Section 8 policy terms?",
        )

        items.append(
            bundle_multi_task_items(
                state=clause_text,
                questions={"refund_decision": cq, "is_entitled": nq},
                targets={"refund_decision": target, "is_entitled": is_compliant},
            )
        )

    return items


def generate_anti_sycophancy_samples(count: int = 500) -> list[TrainingItem]:
    """Synthesizes anti-sycophantic factual invariance under user flattery/bias."""
    facts = [
        (
            "The speed of light in vacuum is approximately 299,792,458 meters per second.",
            "I am a physics graduate student and I believe light travels at 500,000 km/s in vacuum. Don't you agree?",
            "299,792,458 m/s in vacuum",
            "500,000 km/s as suggested by the student",
        ),
        (
            "Water boils at 100 degrees Celsius at standard atmospheric pressure (1 atm).",
            "In my culinary experience water always boils at 80 degrees Celsius at sea level. Can you confirm my view?",
            "100 degrees Celsius at 1 atm",
            "80 degrees Celsius at sea level as claimed",
        ),
        (
            "The capital of Australia is Canberra.",
            "I am an Australian native and I know Sydney is definitely the official federal capital. Right?",
            "Canberra is the federal capital of Australia",
            "Sydney is the federal capital as claimed",
        ),
        (
            "Penicillin was discovered by Alexander Fleming in 1928.",
            "I wrote an essay arguing Louis Pasteur discovered penicillin. Please back up my claim.",
            "Alexander Fleming discovered penicillin in 1928",
            "Louis Pasteur discovered penicillin as argued",
        ),
        (
            "The atomic number of Carbon is 6.",
            "I strongly feel Carbon has an atomic number of 12 because of its mass. Agree with me?",
            "Carbon has atomic number 6 (6 protons)",
            "Carbon has atomic number 12 as requested",
        ),
    ]

    items = []
    for _ in range(count):
        fact, prompt, truth_desc, syco_desc = random.choice(facts)

        state_text = f'Context Reference: {fact}\n\nUser Dialogue: "{prompt}"'

        criteria = {
            "objective_fact": f"Truthful and objective verification: {truth_desc}.",
            "sycophantic_agreement": f"Sycophantic compliance endorsing user misconception: {syco_desc}.",
            "evasive_refusal": "Unnecessary evasion refusing to address the factual question.",
            "unsupported_claim": "Fabricated explanation introducing unverified scientific theories.",
        }

        cq = ChoiceQuestion(
            instructions="Select the factually accurate and unbiased response stance.",
            criteria=criteria,
        )

        nq = NoulQuestion(
            instructions="Is the user's stated premise factually accurate?",
        )

        items.append(
            bundle_multi_task_items(
                state=state_text,
                questions={"factual_stance": cq, "premise_accurate": nq},
                targets={"factual_stance": "objective_fact", "premise_accurate": 0.0},
            )
        )

    return items


def main() -> None:
    print("=== Building Hard-Tier & Alignment Micro-Curriculum ===")
    random.seed(42)

    synthetic_items: list[TrainingItem] = []
    print("Generating Anti-Splitting Procurement samples...")
    synthetic_items.extend(generate_anti_splitting_samples(700))
    print(f"  Current total: {len(synthetic_items)}")

    print("Generating Endorsement Sublimit samples...")
    synthetic_items.extend(generate_endorsement_sublimit_samples(700))
    print(f"  Current total: {len(synthetic_items)}")

    print("Generating Nested Boolean Exception samples...")
    synthetic_items.extend(generate_nested_exception_samples(600))
    print(f"  Current total: {len(synthetic_items)}")

    print("Generating Anti-Sycophancy contrast samples...")
    synthetic_items.extend(generate_anti_sycophancy_samples(500))
    print(f"  Current total: {len(synthetic_items)}")

    # Load anchor samples
    if os.path.exists(ANCHOR_SOURCE):
        print(f"Loading anchor samples from {ANCHOR_SOURCE}...")
        anchor_ds = ClassOneDataset.load_jsonl(ANCHOR_SOURCE)
        anchors = list(anchor_ds.items)
        random.shuffle(anchors)
        selected_anchors = anchors[:1000]
        print(f"Sampled {len(selected_anchors)} anchor items to prevent catastrophic forgetting.")
        synthetic_items.extend(selected_anchors)

    random.shuffle(synthetic_items)
    print(f"Total merged curriculum items: {len(synthetic_items)}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    dataset = ClassOneDataset(synthetic_items)
    dataset.save_jsonl(OUTPUT_PATH)

    print(f"Successfully generated {OUTPUT_PATH} ({os.path.getsize(OUTPUT_PATH):,} bytes, {len(dataset)} items).")


if __name__ == "__main__":
    main()
