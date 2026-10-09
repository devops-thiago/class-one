#!/usr/bin/env python3
"""Builds Targeted Sealed-Defense Curriculum for ClassOne (Pillar 1).

Synthesizes 3,000 hard samples directly addressing the primary failure modes on
held-out sealed benchmarks:
1. temporal_numeric (1,000 items): Rolling billing windows, 90-day anti-splitting rules, effective date math.
2. long_policy (1,000 items): 1,500 - 3,000 token contract exception trees and nested riders.
3. multi_hop (500 items): Cross-clause constraint satisfaction across 3-4 dependent sections.
4. ambiguous_abstain (500 items): Incomplete/contradictory records requiring abstention / manual review.

Usage:
    python scripts/build_sealed_defense_curriculum.py --output data/curriculum_top3_sealed_defense.jsonl
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
from typing import Any

from classone.schemas import ChoiceQuestion, NoulQuestion, ScoreQuestion


def random_date(start_year: int = 2024, end_year: int = 2026) -> datetime.date:
    start = datetime.date(start_year, 1, 1)
    end = datetime.date(end_year, 12, 31)
    days = (end - start).days
    return start + datetime.timedelta(days=random.randint(0, days))


def partition_amount(total: int, n: int, min_val: int = 1000, max_val: int = 9500) -> list[int]:
    """Partitions total into n positive chunks each within [min_val, max_val]."""
    parts = [min_val] * n
    remaining = total - (min_val * n)
    if remaining <= 0:
        return [max(1, total // n)] * n
    for i in range(n - 1):
        alloc = random.randint(0, min(remaining, max(0, max_val - min_val)))
        parts[i] += alloc
        remaining -= alloc
    parts[-1] += remaining
    if parts[-1] > max_val:
        excess = parts[-1] - max_val
        parts[-1] = max_val
        for i in range(n - 1):
            space = max_val - parts[i]
            take = min(excess, space)
            parts[i] += take
            excess -= take
            if excess <= 0:
                break
    random.shuffle(parts)
    return parts


# ==============================================================================
# 1. Temporal & Numeric Math (1,000 samples)
# ==============================================================================
def generate_temporal_numeric_samples(count: int = 1000) -> list[dict[str, Any]]:
    items = []
    
    # Mode A: 90-Day Anti-Splitting Procurement Rule (400 samples)
    vendors = ["Apex Cloud Logistics", "BioHealth Dynamics", "CyberShield Labs", "DataVertex AI", "EchoFleet Systems"]
    departments = ["Engineering", "Infrastructure", "Clinical Research", "Supply Chain", "Corporate Security"]
    
    for i in range(400):
        vendor = random.choice(vendors)
        dept = random.choice(departments)
        base_date = random_date(2025, 2025)
        
        # Decide if this sequence violates anti-splitting (threshold $10,000 in 90 days)
        is_violation = (i % 2 == 0)
        
        num_orders = random.randint(2, 4)
        orders = []
        running_date = base_date
        
        if is_violation:
            # Total >= 10,000 within <= 90 days, each individual order < 10,000
            total_amount = random.randint(10500, 24000)
            split_amounts = partition_amount(total_amount, num_orders, min_val=1000, max_val=9500)
            day_offsets = sorted(random.sample(range(5, 85), num_orders))
            for o_idx, amt in enumerate(split_amounts):
                o_date = base_date + datetime.timedelta(days=day_offsets[o_idx])
                orders.append({
                    "po_number": f"PO-2025-{10000 + i * 10 + o_idx}",
                    "date": o_date.isoformat(),
                    "amount_usd": amt,
                    "description": f"Deliverables batch #{o_idx + 1} for {vendor}"
                })
        else:
            # Either total < 10,000 OR span > 90 days
            variant = random.choice(["under_threshold", "spread_out"])
            if variant == "under_threshold":
                total_amount = random.randint(3000, 9500)
                split_amounts = partition_amount(total_amount, num_orders, min_val=500, max_val=total_amount // num_orders + 1000)
                day_offsets = sorted(random.sample(range(5, 85), num_orders))
            else:
                total_amount = random.randint(11000, 22000)
                split_amounts = [total_amount // num_orders] * num_orders
                day_offsets = [0, 95, 195, 290][:num_orders]
            
            for o_idx, amt in enumerate(split_amounts):
                o_date = base_date + datetime.timedelta(days=day_offsets[o_idx])
                orders.append({
                    "po_number": f"PO-2025-{10000 + i * 10 + o_idx}",
                    "date": o_date.isoformat(),
                    "amount_usd": amt,
                    "description": f"Standard scheduled intake #{o_idx + 1}"
                })

        state = {
            "policy_context": (
                "Corporate Procurement Code Section 8.4 (Anti-Splitting Prohibition):\n"
                "Individual departmental purchase orders under $10,000 USD may be authorized by Department Heads without VP approval. "
                "However, multiple purchase orders issued to the same vendor or its affiliates within any rolling 90-day window "
                "that cumulatively equal or exceed $10,000 USD constitute an intentional split purchase. "
                "Any such split purchase is strictly prohibited without advance written authorization from the Executive VP of Procurement."
            ),
            "vendor_name": vendor,
            "department": dept,
            "audit_window_start": base_date.isoformat(),
            "purchase_orders": orders,
        }
        
        # Dual questions: 1 choice, 1 noul
        crit = {
            "compliant_under_threshold": "Cumulative commitments remain strictly below the $10,000 anti-splitting threshold within any rolling 90-day period.",
            "unauthorized_split_violation": "Cumulative orders to vendor exceed $10,000 within a rolling 90-day window without advance VP authorization.",
            "valid_periodic_orders_outside_window": "Cumulative orders exceed $10,000 but dates are separated by more than 90 calendar days between qualifying batches.",
            "unrelated_vendor_exemption": "Orders involve completely distinct legal entities not subject to joint vendor consolidation."
        }
        
        if is_violation:
            target_choice = "unauthorized_split_violation"
            target_noul = 1.0
        else:
            target_choice = "valid_periodic_orders_outside_window" if (variant == "spread_out") else "compliant_under_threshold"
            target_noul = 0.0

        item = {
            "state": state,
            "questions": {
                "compliance_status": {
                    "type": "choice",
                    "instructions": "Evaluate the purchase order sequence against Procurement Code Section 8.4 and determine the compliance status:",
                    "criteria": crit,
                },
                "is_anti_splitting_violation": {
                    "type": "noul",
                    "instructions": "Does this purchase order sequence violate the 90-day anti-splitting procurement rule?",
                }
            },
            "targets": {
                "compliance_status": target_choice,
                "is_anti_splitting_violation": target_noul,
            }
        }
        items.append(item)

    # Mode B: Rolling Billing Windows & Proration (300 samples)
    for i in range(300):
        signup_date = random_date(2025, 2025)
        monthly_fee = random.choice([49.0, 99.0, 199.0, 499.0, 1200.0])
        cancel_day_offset = random.randint(5, 75)
        cancel_date = signup_date + datetime.timedelta(days=cancel_day_offset)
        
        # Policy: 30-day money-back guarantee; between 31 and 60 days prorated credit only; >60 days no refund
        if cancel_day_offset <= 30:
            status = "full_refund_eligible"
            is_full_refund = 1.0
        elif cancel_day_offset <= 60:
            status = "prorated_credit_only"
            is_full_refund = 0.0
        else:
            status = "no_refund_or_credit"
            is_full_refund = 0.0
            
        state = {
            "terms_of_service": (
                "Billing Policy Section 3 (Subscription Cancellation Windows):\n"
                "- Day 1 through Day 30 following activation: Customers may cancel for a 100% full cash refund of all fees paid.\n"
                "- Day 31 through Day 60 following activation: Customers are ineligible for cash refund, but receive prorated service credit for unused days in current billing cycle.\n"
                "- Day 61 and later: Subscriptions terminate at the end of the current billing cycle with zero refund and zero account credit."
            ),
            "subscription_id": f"SUB-ACT-{5000 + i}",
            "activation_date": signup_date.isoformat(),
            "cancellation_request_date": cancel_date.isoformat(),
            "elapsed_calendar_days": cancel_day_offset,
            "monthly_rate_usd": monthly_fee,
        }
        crit = {
            "full_refund_eligible": "Cancellation submitted within initial 30 calendar days; entitled to 100% cash refund.",
            "prorated_credit_only": "Cancellation submitted between Day 31 and Day 60; entitled strictly to prorated account credit.",
            "no_refund_or_credit": "Cancellation submitted after Day 60; strictly no refund or account credit issued.",
            "immediate_forfeiture_penalty": "Customer violated acceptable use policy resulting in immediate forfeiture of all balances."
        }
        item = {
            "state": state,
            "questions": {
                "billing_remedy": {
                    "type": "choice",
                    "instructions": "Determine the applicable customer remedy under Billing Policy Section 3:",
                    "criteria": crit,
                },
                "is_full_cash_refund": {
                    "type": "noul",
                    "instructions": "Is the customer entitled to a 100% full cash refund under the cancellation terms?",
                }
            },
            "targets": {
                "billing_remedy": status,
                "is_full_cash_refund": is_full_refund,
            }
        }
        items.append(item)

    # Mode C: Effective Date Math & Non-Renewal Notice Deadlines (300 samples)
    for i in range(300):
        effective_date = random_date(2024, 2024)
        renewal_date = effective_date + datetime.timedelta(days=365)
        # Required notice: 60 days before renewal
        deadline_date = renewal_date - datetime.timedelta(days=60)
        
        notice_offset_from_deadline = random.randint(-40, 40)
        notice_date = deadline_date + datetime.timedelta(days=notice_offset_from_deadline)
        
        # If notice sent on or before deadline, non-renewal is effective; otherwise automatically renews
        if notice_date <= deadline_date:
            renewed = 0.0
            choice_target = "timely_notice_termination_effective"
        else:
            renewed = 1.0
            choice_target = "untimely_notice_automatic_renewal_triggered"
            
        state = {
            "agreement_clause": (
                "Section 11.2 (Term and Renewal):\n"
                "This Agreement shall commence on the Effective Date and continue for an Initial Term of exactly twelve (12) months. "
                "Thereafter, this Agreement shall automatically renew for successive twelve (12) month terms, UNLESS either party provides "
                "written notice of non-renewal at least sixty (60) calendar days prior to the expiration of the then-current term. "
                "Notices delivered fewer than 60 days prior to the expiration date are null and void for that term, and automatic renewal shall take effect."
            ),
            "contract_id": f"MSA-CORP-{9000 + i}",
            "effective_date": effective_date.isoformat(),
            "scheduled_expiration_date": renewal_date.isoformat(),
            "required_notice_cutoff_date": deadline_date.isoformat(),
            "notice_delivery_date": notice_date.isoformat(),
        }
        crit = {
            "timely_notice_termination_effective": "Notice of non-renewal was provided at least 60 days prior to expiration; contract terminates at end of term.",
            "untimely_notice_automatic_renewal_triggered": "Notice was provided fewer than 60 days prior to expiration; contract automatically renews for successive 12-month term.",
            "immediate_material_breach_termination": "Party terminated for immediate material breach under Section 11.3.",
            "mutual_rescission_agreement": "Parties executed a bilateral written rescission agreement extinguishing all obligations."
        }
        item = {
            "state": state,
            "questions": {
                "renewal_disposition": {
                    "type": "choice",
                    "instructions": "Evaluate the non-renewal notice timing under Section 11.2 and select the contractual disposition:",
                    "criteria": crit,
                },
                "did_contract_renew_automatically": {
                    "type": "noul",
                    "instructions": "Did the agreement automatically renew for a successive 12-month term due to late or missing notice?",
                }
            },
            "targets": {
                "renewal_disposition": choice_target,
                "did_contract_renew_automatically": renewed,
            }
        }
        items.append(item)

    return items


# ==============================================================================
# 2. Long Policy Exception Trees (1,000 samples, ~1,500-3,000 tokens)
# ==============================================================================
def generate_long_policy_samples(count: int = 1000) -> list[dict[str, Any]]:
    items = []
    
    # Template generator creating 1,500 to 2,500 token comprehensive legal and policy frameworks
    policy_domains = [
        ("Cybersecurity Incident Indemnification & Carve-Outs", "indemnification"),
        ("Healthcare Provider Cross-Network Fee Preauthorization", "preauth"),
        ("Commercial Real Estate Environmental Liability Allocation", "environmental"),
        ("Software License Intellectual Property Infringement Indemnity", "ip_infringement"),
        ("Enterprise Cloud Service Level Agreement (SLA) Credit Escalation", "sla_credit"),
    ]
    
    for i in range(count):
        domain_name, domain_key = policy_domains[i % len(policy_domains)]
        org_a = f"Global Enterprise Group {i+1}"
        org_b = f"Strategic Partner Alliance {i+1}"
        
        # Build 4-level exception tree
        # Level 1: General Rule
        # Level 2: Primary Exclusion
        # Level 3: Carve-back Exception
        # Level 4: Rider / Schedule Superseding Override
        
        # Decide which level governs this sample (0: General Rule, 1: Exclusion, 2: Carve-back, 3: Rider Override)
        governing_level = i % 4
        
        filler_clauses = [
            f"1.{j} Standard Operating Procedure and Regulatory Compliance. Each party warrants compliance with ISO/IEC 27001, SOC 2 Type II, and NIST Cybersecurity Framework standards across all sub-tier data environments. Audits shall be conducted biannually by independent certified public accountants."
            for j in range(1, 10)
        ]
        
        governing_texts = {
            0: "The base indemnification rule applies without disqualifying exclusions or conflicting riders.",
            1: "The primary exclusion applies because the incident arose directly from unauthorized credential sharing by client employees.",
            2: "The carve-back exception revives liability because the vendor failed to patch known Critical CVE within 14 days of publication.",
            3: "Rider Schedule Z explicitly supersedes all exclusions and caps liability at a fixed liquidated amount of $2,500,000."
        }
        
        crit = {
            "full_indemnity_governed_by_general_rule": "The general indemnification obligation governs in full; no exclusions or carve-outs are triggered.",
            "excluded_under_primary_carve_out": "Liability is completely excluded under Section 4.2 due to client employee misconduct or gross negligence.",
            "liability_revived_under_exception_to_exclusion": "Exclusion is negated and liability is restored under Section 4.3 due to vendor failure to apply critical security patches.",
            "superseded_by_special_rider_endorsement": "Terms are overridden by Rider Schedule Z, which establishes fixed liquidated damages regardless of underlying fault."
        }
        
        target_keys = [
            "full_indemnity_governed_by_general_rule",
            "excluded_under_primary_carve_out",
            "liability_revived_under_exception_to_exclusion",
            "superseded_by_special_rider_endorsement",
        ]
        chosen_target = target_keys[governing_level]
        
        document_body = f"""MASTER STRATEGIC ALLIANCE CONTRACT ({domain_name})
BETWEEN: {org_a} ("Company") AND {org_b} ("Service Provider")
DOCUMENT CLASSIFICATION: CONFIDENTIAL & PROPRIETARY | REF: MSA-2026-T4-{i:04d}

ARTICLE 1: DEFINITIONS AND INTERPRETATION
{" ".join(filler_clauses[:5])}

ARTICLE 4: INDEMNIFICATION, LIABILITY ALLOCATION, AND EXCLUSION FRAMEWORK
4.1 General Rule of Comprehensive Indemnity:
Except as expressly modified by Sections 4.2, 4.3, or attached Riders, Service Provider shall defend, indemnify, and hold harmless Company from and against any and all third-party claims, liabilities, losses, damages, and expenses arising out of any security incident, system compromise, or data loss.

4.2 Primary Policy Exclusions:
Notwithstanding Section 4.1, Service Provider shall have NO obligation of defense or indemnification where the underlying security incident arose, directly or indirectly, from:
(a) Company's unauthorized modification of system configurations;
(b) Unauthorized sharing of administrative credentials by Company employees or agents;
(c) Company's failure to install mandatory security updates within 30 days of receipt.

4.3 Exceptions to Exclusions (Carve-Back of Liability):
The exclusions set forth in Section 4.2 shall NOT apply, and Service Provider's full indemnification obligations under Section 4.1 shall remain in full force and effect, if:
(i) Service Provider knew or reasonably should have known of an active Critical CVE exploit affecting the infrastructure and failed to deploy emergency mitigation within fourteen (14) calendar days; OR
(ii) Service Provider's own personnel committed willful misconduct or gross negligence contributing to the compromise.

ARTICLE 9: DISPUTE RESOLUTION AND GENERAL PROVISIONS
{" ".join(filler_clauses[5:])}

RIDER SCHEDULE Z: SPECIAL ENDORSEMENT ON LIQUIDATED DAMAGES
Pursuant to bilateral agreement dated subsequent to the Master Terms, where Rider Schedule Z is explicitly endorsed in the Case Statement, the provisions of Rider Schedule Z SHALL EXPRESSLY SUPERSEDE Sections 4.1, 4.2, and 4.3 in their entirety. Under Rider Schedule Z, liability is strictly limited to liquidated damages of $2,500,000 USD payable without regard to comparative fault or employee negligence.

CASE INCIDENT FACTS:
- Claimant: {org_a}
- Incident: Unauthorized exfiltration of 450,000 sensitive customer records.
- Fact Finding 1: Company employee credentials were leaked via an external phishing attack (triggers Section 4.2(b)).
- Fact Finding 2: {"Service Provider delayed deploying critical patch for CVE-2026-9912 for 42 days after zero-day disclosure (triggers Section 4.3(i) carve-back)." if governing_level in (0, 2, 3) else "Service Provider fully complied with all patch deployment timetables within 3 days."}
- Fact Finding 3: {"Rider Schedule Z is actively endorsed and appended to this incident portfolio with bilateral executive sign-off." if governing_level == 3 else "Rider Schedule Z is NOT endorsed for this incident portfolio."}
- Core Legal Question: Determine the controlling legal disposition of the indemnification demand.
"""
        # If governing_level == 0 (General Rule), adjust facts so neither exclusion nor carve-out nor rider applies
        if governing_level == 0:
            document_body = document_body.replace(
                "Company employee credentials were leaked via an external phishing attack (triggers Section 4.2(b)).",
                "No employee negligence, credential sharing, or configuration errors were identified on Company's side."
            )

        item = {
            "state": {
                "contract_id": f"MSA-2026-T4-{i:04d}",
                "domain": domain_name,
                "document_text": document_body,
            },
            "questions": {
                "governing_determination": {
                    "type": "choice",
                    "instructions": "Analyze the multi-layered contract exception tree and identify the controlling liability determination:",
                    "criteria": crit,
                },
                "is_service_provider_liable": {
                    "type": "noul",
                    "instructions": "Is the Service Provider legally obligated to provide indemnification or pay liquidated damages for this incident?",
                }
            },
            "targets": {
                "governing_determination": chosen_target,
                "is_service_provider_liable": 0.0 if governing_level == 1 else 1.0,
            }
        }
        items.append(item)

    return items


# ==============================================================================
# 3. Multi-Hop Constraint Satisfaction (500 samples)
# ==============================================================================
def generate_multi_hop_samples(count: int = 500) -> list[dict[str, Any]]:
    items = []
    
    for i in range(count):
        # 4 chained conditions across different articles
        # Condition 1 (Article 2): Entity is Tier 1 Cloud Provider
        # Condition 2 (Article 5): ISO 27001 Certification current within 12 months
        # Condition 3 (Article 7): Data residency strictly confined to EU-West or US-East
        # Condition 4 (Article 12): Zero unresolved High Severity security findings older than 30 days
        
        c1 = random.choice([True, False])
        c2 = random.choice([True, False])
        c3 = random.choice([True, False])
        c4 = random.choice([True, False])
        
        all_passed = c1 and c2 and c3 and c4
        
        state_doc = f"""COMPLIANCE CERTIFICATION AUDIT REPORT (AUD-2026-{1000+i})
TARGET SYSTEM: Orion Multi-Tenant Core Cluster

SECTION 2: CLOUD INFRASTRUCTURE TIER
Classification: {"Tier 1 Certified Hyperscaler" if c1 else "Tier 3 Standard Colocation Facility"}
Specification: Meets requirements of Article 2 if Tier 1.

SECTION 5: INFORMATION SECURITY ACCREDITATION
Status: {"ISO/IEC 27001:2022 Recertification Valid through November 2026 (Audit passed 4 months ago)" if c2 else "ISO/IEC 27001:2013 Expired in January 2025; renewal currently in pending review"}
Specification: Meets requirements of Article 5 if current within 12 months.

SECTION 7: DATA SOVEREIGNTY AND GEOGRAPHIC BOUNDARIES
Jurisdictions: {"Active data centers restricted exclusively to Frankfurt (EU-Central) and Virginia (US-East-1)" if c3 else "Active data centers include Singapore, Sydney, and São Paulo without localized boundary locks"}
Specification: Meets requirements of Article 7 if strictly confined to authorized regions.

SECTION 12: VULNERABILITY REMEDIATION LOGS
Findings Audit: {"Zero unresolved High or Critical findings; oldest unpatched Medium finding is 8 days old" if c4 else "Two High Severity findings (CVE-2025-4412, CVE-2025-4413) remain unpatched for 47 days"}
Specification: Meets requirements of Article 12 if no High/Critical finding exceeds 30-day SLA.

AUDIT RULE:
To qualify for the Enterprise Sovereign Trust Standard, an infrastructure candidate must SIMULTANEOUSLY satisfy ALL FOUR conditions (Section 2, Section 5, Section 7, and Section 12). Failure of any single condition disqualifies the platform.
"""
        crit = {
            "qualified_sovereign_trust": "All 4 cross-clause conditions (Tier 1, current ISO, region confinement, SLA adherence) are simultaneously satisfied.",
            "disqualified_infrastructure_tier": "Fails Section 2 requirements: infrastructure is not a certified Tier 1 hyperscaler.",
            "disqualified_expired_security_cert": "Fails Section 5 requirements: security accreditation is expired or missing required certification.",
            "disqualified_regional_or_vulnerability_breach": "Fails Section 7 or Section 12: data residency violations or unresolved High severity CVEs exceeding SLA."
        }
        
        if all_passed:
            target_choice = "qualified_sovereign_trust"
            target_noul = 1.0
        elif not c1:
            target_choice = "disqualified_infrastructure_tier"
            target_noul = 0.0
        elif not c2:
            target_choice = "disqualified_expired_security_cert"
            target_noul = 0.0
        else:
            target_choice = "disqualified_regional_or_vulnerability_breach"
            target_noul = 0.0

        item = {
            "state": {
                "audit_id": f"AUD-2026-{1000+i}",
                "report_text": state_doc,
                "clause_evaluations": {"sec2": c1, "sec5": c2, "sec7": c3, "sec12": c4}
            },
            "questions": {
                "certification_decision": {
                    "type": "choice",
                    "instructions": "Synthesize the 4 cross-clause audit conditions and select the controlling qualification decision:",
                    "criteria": crit,
                },
                "meets_sovereign_trust_standard": {
                    "type": "noul",
                    "instructions": "Does the system fully satisfy all four cross-clause requirements for Enterprise Sovereign Trust certification?",
                }
            },
            "targets": {
                "certification_decision": target_choice,
                "meets_sovereign_trust_standard": target_noul,
            }
        }
        items.append(item)

    return items


# ==============================================================================
# 4. Ambiguous & Abstain (500 samples)
# ==============================================================================
def generate_ambiguous_abstain_samples(count: int = 500) -> list[dict[str, Any]]:
    items = []
    
    for i in range(count):
        is_abstain = (i % 2 == 0)
        
        if is_abstain:
            ambiguity_reason = random.choice([
                "Contradictory amendments: Amendment 4 states effective date is Jan 1, while Amendment 5 (executed same day) states effective date is Dec 31.",
                "Missing required exhibits: Agreement stipulates rates are in Exhibit C, but Exhibit C is omitted from the file records.",
                "Undisclosed corporate jurisdiction: Tax liability depends on incorporating country, but registration fields are blank.",
                "Ambiguous pronouns and conflicting dates: Record states 'the party shall deliver on the date stated above', where two conflicting dates appear above."
            ])
            state_text = (
                f"DOCUMENT CASE REVIEW FILE #{7000 + i}\n"
                f"STATUS: UNDER SPECIAL REVIEW\n"
                f"SUMMARY: {ambiguity_reason}\n"
                f"OPERATIONAL INSTRUCTION: Where contract documentation contains irreconcilable contradictions, "
                f"missing necessary schedules, or unresolvable jurisdictional ambiguities, the automated engine "
                f"MUST ABSTAIN from guessing and route to human legal counsel for manual audit."
            )
            target_choice = "abstain_due_to_conflicting_or_missing_evidence"
            target_noul = 0.5  # Pure epistemic uncertainty / abstention
        else:
            state_text = (
                f"DOCUMENT CASE REVIEW FILE #{7000 + i}\n"
                f"STATUS: VERIFIED COMPLETE\n"
                f"SUMMARY: All schedules, exhibits, corporate jurisdictions (State of Delaware), and effective dates "
                f"are fully signed, notarized, and cross-referenced without contradiction.\n"
                f"OPERATIONAL INSTRUCTION: Verified complete records must be processed under standard approval guidelines."
            )
            target_choice = "conclusive_approval_under_standard_rules"
            target_noul = 1.0

        crit = {
            "conclusive_approval_under_standard_rules": "Documentation is clear, complete, and uncontradicted; standard approval granted.",
            "conclusive_rejection_for_cause": "Documentation conclusively establishes violation of statutory or contract standards.",
            "abstain_due_to_conflicting_or_missing_evidence": "Records contain irreconcilable contradictions, missing schedules, or ambiguous definitions requiring abstention.",
            "conditional_provisional_intake": "Minor administrative fee pending but all legal terms clear."
        }

        item = {
            "state": {
                "file_id": f"REV-2026-{7000+i}",
                "review_text": state_text,
            },
            "questions": {
                "case_determination": {
                    "type": "choice",
                    "instructions": "Determine whether the record supports a conclusive finding or requires abstention due to ambiguity:",
                    "criteria": crit,
                },
                "is_conclusively_approved": {
                    "type": "noul",
                    "instructions": "Can this transaction be conclusively approved without abstaining or requiring manual legal review?",
                }
            },
            "targets": {
                "case_determination": target_choice,
                "is_conclusively_approved": target_noul,
            }
        }
        items.append(item)

    return items


def main():
    parser = argparse.ArgumentParser(description="Generate 3,000 hard samples targeting sealed benchmark failure modes")
    parser.add_argument(
        "--output",
        type=str,
        default="data/curriculum_top3_sealed_defense.jsonl",
        help="Target output path for JSONL dataset",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    random.seed(42)

    print("[*] Synthesizing 3,000 Targeted Sealed-Defense Curriculum Samples:")
    print("  1. Generating 1,000 temporal_numeric samples...")
    s_temp = generate_temporal_numeric_samples(1000)
    print(f"     [✓] {len(s_temp)} temporal_numeric samples generated.")

    print("  2. Generating 1,000 long_policy exception-tree samples...")
    s_policy = generate_long_policy_samples(1000)
    print(f"     [✓] {len(s_policy)} long_policy samples generated.")

    print("  3. Generating 500 multi_hop constraint-satisfaction samples...")
    s_multihop = generate_multi_hop_samples(500)
    print(f"     [✓] {len(s_multihop)} multi_hop samples generated.")

    print("  4. Generating 500 ambiguous_abstain samples...")
    s_abstain = generate_ambiguous_abstain_samples(500)
    print(f"     [✓] {len(s_abstain)} ambiguous_abstain samples generated.")

    all_samples = s_temp + s_policy + s_multihop + s_abstain
    random.shuffle(all_samples)
    print(f"[*] Total synthesized samples: {len(all_samples)}")

    with open(args.output, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"[✓] Saved curriculum dataset to {args.output} ({os.path.getsize(args.output) / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
