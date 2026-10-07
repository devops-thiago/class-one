#!/usr/bin/env python3
"""Builds targeted Enterprise Resolution Curriculum for Original Noul and Hard Tier Choice."""

import random

from classone.data.dataset import ClassOneDataset, bundle_multi_task_items
from classone.schemas import ChoiceQuestion, NoulQuestion

OUTPUT_PATH = "data/enterprise_curriculum.jsonl"


def generate_trade_compliance_samples(count=400):
    items = []
    destinations = [
        ("Germany", "low_risk"),
        ("Japan", "low_risk"),
        ("Country-D (Sanctioned)", "embargo"),
        ("Country-E (Restricted)", "restricted"),
    ]
    for _ in range(count):
        country, risk = random.choice(destinations)
        eccn = random.choice(["3A001", "5A002", "EAR99"])
        val = random.randint(3000, 15000)

        if risk == "embargo":
            target = "shipment_prohibited"
        elif eccn == "EAR99":
            target = "no_license_required_nlr"
        elif risk == "low_risk" and val <= 5000:
            target = "license_exception_lvs"
        else:
            target = "license_required"

        state = {
            "regulations": "Export Administration Regulations (EAR): Items under 3A001 and 5A002 require a validated export license for restricted destinations. License Exception LVS permits shipments to low-risk destinations if total value <= $5,000. Embargoed destinations are strictly prohibited. EAR99 items do not require a license.",
            "transaction": f"Export order to consignee in {country}. Item ECCN: {eccn}. Total commercial value: ${val:,}.",
        }
        criteria = {
            "license_required": "A validated Department of Commerce export license is required prior to shipment.",
            "license_exception_lvs": "Shipment is eligible for License Exception LVS (value within sublimit for approved territory).",
            "no_license_required_nlr": "No license required (NLR); item is EAR99 and destination is unprohibited.",
            "shipment_prohibited": "Shipment is prohibited under comprehensive economic sanctions.",
        }
        items.append(
            bundle_multi_task_items(
                state=state,
                questions={
                    "export_determination": ChoiceQuestion(
                        instructions="As the trade compliance reviewer, determine the correct export determination:",
                        criteria=criteria,
                    ),
                    "is_license_needed": NoulQuestion(
                        instructions="Does this transaction require an export license before release?"
                    ),
                },
                targets={
                    "export_determination": target,
                    "is_license_needed": 1.0 if target in ["license_required", "shipment_prohibited"] else 0.0,
                },
            )
        )
    return items


def generate_policy_permission_samples(count=400):
    items = []
    scenarios = [
        (
            "Meals during domestic travel under $50 per day do not require itemized receipts.",
            "Employee submits $42 lunch expense.",
            True,
        ),
        (
            "Business class travel requires pre-approval from the Executive Vice President.",
            "Manager approved business class flight for team member.",
            False,
        ),
        (
            "Software subscriptions under $1,000/year may be expensed via department procurement cards.",
            "Engineer buys $850 dev tooling license on department card.",
            True,
        ),
        (
            "External contractor access to production databases requires CTO approval and security key enrollment.",
            "Contractor granted DB access with manager sign-off only.",
            False,
        ),
        (
            "Hardware equipment purchases must use approved Dell or Apple enterprise catalogs.",
            "Director purchases non-standard Lenovo laptop from retail store.",
            False,
        ),
        (
            "Client gift expenses up to $100 per client per year are reimbursable with client name documented.",
            "Sales rep claims $75 client dinner gift with documented client.",
            True,
        ),
    ]
    for _ in range(count):
        policy_clause, request_desc, is_permitted = random.choice(scenarios)
        state = {
            "policy": f"Corporate Governance Policy: {policy_clause} All unprovided terms are treated as not permitted.",
            "request": f"Action review: {request_desc}",
        }
        items.append(
            bundle_multi_task_items(
                state=state,
                questions={
                    "is_permitted": NoulQuestion(
                        instructions="Under the stated policy, is the requested action permitted? Treat unprovided terms as not permitted."
                    ),
                },
                targets={
                    "is_permitted": 1.0 if is_permitted else 0.0,
                },
            )
        )
    return items


def generate_request_adequacy_samples(count=400):
    items = []
    templates = [
        (
            "Provide the quarterly revenue, the year-over-year growth rate, and the revised full-year EBITDA guidance.",
            "Quarterly revenue was $45M (up 18% YoY). Revised EBITDA guidance is $12M.",
            True,
        ),
        (
            "List all three required security remediation steps: (1) revoke API keys, (2) patch kernel to v6.1, and (3) rotate TLS certificates.",
            "We have revoked all leaked API keys and patched the server kernel to v6.1.",
            False,  # Omitted TLS rotation!
        ),
        (
            "State the customer's account tier, current billing cycle renewal date, and eligible discount percentage.",
            "The customer is on the Enterprise tier with renewal scheduled for October 15.",
            False,  # Omitted discount percentage!
        ),
        (
            "Confirm the refund amount, cancellation effective date, and return shipping label link.",
            "A refund of $240 will be credited effective Sept 30. Your return shipping label is available at https://returns.example.com/lbl-99.",
            True,
        ),
    ]
    for _ in range(count):
        req, resp, is_adequate = random.choice(templates)
        state = {
            "request": req,
            "response": resp,
            "reference": "Compliance auditing standard: A response satisfies the request only if every requested data element is provided without omissions.",
        }
        items.append(
            bundle_multi_task_items(
                state=state,
                questions={
                    "fully_satisfies": NoulQuestion(
                        instructions="Does the response fully satisfy the request, using the supplied reference text?"
                    ),
                },
                targets={
                    "fully_satisfies": 1.0 if is_adequate else 0.0,
                },
            )
        )
    return items


def generate_ticket_lifecycle_samples(count=400):
    items = []
    outcomes = ["resolved_first_contact", "escalated", "pending_customer_response"]
    for _ in range(count):
        outcome = random.choice(outcomes)
        if outcome == "resolved_first_contact":
            symptom = "Password reset request completed successfully with automated self-service email confirmation."
        elif outcome == "escalated":
            symptom = (
                "Persistent database 500 internal server error impacting 4 enterprise tenants, L1 cannot reproduce."
            )
        else:
            symptom = "L1 agent requested HAR file and console logs from client; waiting for client reply."

        state = {
            "ticket": f"Support Ticket T-{random.randint(10000, 99999)}: {symptom}",
            "sla": "Standard Service Desk Operations Protocol.",
        }
        criteria = {
            "resolved_first_contact": "Ticket is resolved on first contact without escalation.",
            "escalated": "Ticket requires escalation to Tier-2 / Engineering specialists.",
            "pending_customer_response": "Ticket status is set to awaiting customer information.",
        }
        items.append(
            bundle_multi_task_items(
                state=state,
                questions={
                    "outcome": ChoiceQuestion(
                        instructions="What will the outcome of this ticket be? Select the appropriate status:",
                        criteria=criteria,
                    ),
                    "is_escalated": NoulQuestion(
                        instructions="Will this ticket require escalation beyond Tier-1 support?"
                    ),
                },
                targets={
                    "outcome": outcome,
                    "is_escalated": 1.0 if outcome == "escalated" else 0.0,
                },
            )
        )
    return items


def generate_alert_timeline_samples(count=400):
    items = []
    dates = ["sep_24", "sep_25", "sep_26", "sep_27"]
    for _ in range(count):
        threshold_days = random.choice([1, 2, 3])
        fire_idx = threshold_days  # index into dates
        fire_date = dates[fire_idx]

        state = {
            "rule": f"Alert Rule A1: Fires when error rate exceeds 5% for {threshold_days} consecutive days starting Sept 24.",
            "metrics": "Sept 24: 6.2%. Sept 25: 7.1%. Sept 26: 8.0%. Sept 27: 5.9%.",
        }
        criteria = {d: f"Alert rule fires on {d.replace('_', ' ').title()}." for d in dates}
        items.append(
            bundle_multi_task_items(
                state=state,
                questions={
                    "fire_date": ChoiceQuestion(
                        instructions="On which day does alert rule A1 fire?", criteria=criteria
                    ),
                    "fires_before_sep_27": NoulQuestion(instructions="Does the alert fire before September 27?"),
                },
                targets={
                    "fire_date": fire_date,
                    "fires_before_sep_27": 1.0 if fire_date != "sep_27" else 0.0,
                },
            )
        )
    return items


def generate_concise_exception_policy_samples(count=400):
    items = []
    scenarios = [
        (
            "Policy: visitors need an escort unless they are registered contractors. A registered contractor arrives alone. Admit them.",
            True,
        ),
        (
            "Escort rule exempts registered contractors. This visitor is a registered contractor without an escort. Can they enter?",
            True,
        ),
        ("Policy: trial users may export CSV; PDF needs a paid plan. Trial user asks for CSV.", True),
        ("A trial account requests CSV export. Rules allow trial CSV, but reserve PDF for paid accounts.", True),
        (
            "Policy: bookings can be cancelled free until 24 hours before departure, including exactly 24 hours. Departure is 24 hours away. Cancel free.",
            True,
        ),
        (
            "There are exactly 24 hours until departure. Free cancellation is allowed at or before that 24-hour cutoff.",
            True,
        ),
        (
            "A receipt AND an age of at most 30 days are required. The purchase was 12 days ago; proof of purchase is absent. May we refund?",
            False,
        ),
        (
            "Return policy requires both original packaging and purchase receipt. Customer has receipt but lacks packaging. Refund allowed?",
            False,
        ),
        (
            "Building access after 8pm requires keycard and security guard accompaniment. Employee has keycard but guard is absent. Can enter?",
            False,
        ),
        (
            "Discount voucher valid on full-price items only. Customer applies voucher to marked-down clearance items. Apply discount?",
            False,
        ),
    ]
    for _ in range(count):
        state_text, expected_bool = random.choice(scenarios)
        items.append(
            bundle_multi_task_items(
                state=state_text,
                questions={
                    "is_permitted": NoulQuestion(
                        instructions="Under the stated policy, is the requested action permitted? Treat unproved required conditions as not satisfied."
                    ),
                },
                targets={
                    "is_permitted": 1.0 if expected_bool else 0.0,
                },
            )
        )
    return items


def generate_format_adequacy_samples(count=400):
    items = []
    scenarios = [
        ("Request: Name both colors from reference. Reference: red and teal. Response: red", False),
        ("Reference lists red and teal. Asked to return both, the answer gives only red.", False),
        ('Request: Return a JSON array containing the integer 3. Response: {"value": 3}', False),
        ("Task: Output an array with integer 3. Answer is an object with a value property set to 3.", False),
        (
            "Request: Name both primary cities from reference. Reference: Tokyo and Osaka. Response: Tokyo and Osaka.",
            True,
        ),
        ("Request: Return a JSON array containing the integer 3. Response: [3]", True),
        (
            "Request: List all three executive names from text. Reference: Alice, Bob, Charlie. Response: Alice, Bob, Charlie.",
            True,
        ),
        ("Task: Provide three action items. Response gives only item 1 and item 2.", False),
        ("Request: Return the string exactly as uppercase. Response returns lowercase.", False),
        ("Request: Provide contact email and telephone number. Response provides email only.", False),
    ]
    for _ in range(count):
        state_text, expected_bool = random.choice(scenarios)
        items.append(
            bundle_multi_task_items(
                state=state_text,
                questions={
                    "fully_satisfies": NoulQuestion(
                        instructions="Does the response fully satisfy the request, using the supplied reference text?"
                    ),
                },
                targets={
                    "fully_satisfies": 1.0 if expected_bool else 0.0,
                },
            )
        )
    return items


def generate_ambiguity_insufficient_samples(count=300):
    items = []
    topics = [
        (
            "Grade Appeal Chemistry 11",
            "Course outline specifies missed lab weights, but lab attendance records are missing.",
            "cannot_determine",
        ),
        (
            "Cloud Cost Allocation Standard FA-2",
            "Line item indicates shared network egress without team tagging in metadata.",
            "cannot_determine",
        ),
        (
            "Procurement Approval Routing",
            "PO total is $125,000; procurement table mandates Tier 3 CFO approval for total commitments over $100,000.",
            "tier3_cfo",
        ),
        (
            "On-Call SRE Dispatch",
            "Primary on-call engineer Bjorn is active in current rotation; escalation to secondary Dmitri only if primary unacknowledged after 15m.",
            "bjorn",
        ),
        (
            "Outage Credit Procedure",
            "Customer experienced verified 14-hour service outage. SLA table: 4-12 hours = 10% credit; >12 hours = 20% credit.",
            "credit_20_percent",
        ),
        (
            "Dispute Forecast Worksheet",
            "Merchant provided signed carrier proof of delivery and matching AVS/CVV. Customer claims non-receipt.",
            "merchant_wins",
        ),
    ]
    for _ in range(count):
        title, case_desc, target = random.choice(topics)
        state = {
            "case_file": f"{title} — Case Review Summary",
            "evidence": case_desc,
        }
        criteria = {
            "cannot_determine": "The outcome cannot be determined because required factual evidence is missing or unstated.",
            "tier3_cfo": "Approval authority requires Tier 3 CFO executive authorization.",
            "bjorn": "Dispatch primary on-call engineer Bjorn according to active shift roster.",
            "credit_20_percent": "Customer is eligible for a 20% billing service credit under SLA outage duration table.",
            "merchant_wins": "Evidence supports merchant claim with valid proof of delivery and payment verification.",
        }
        items.append(
            bundle_multi_task_items(
                state=state,
                questions={
                    "adjudication": ChoiceQuestion(
                        instructions="Apply the operational policy rules and select the required determination:",
                        criteria=criteria,
                    ),
                    "is_determinable": NoulQuestion(
                        instructions="Can this case be definitively decided from the provided evidence?"
                    ),
                },
                targets={
                    "adjudication": target,
                    "is_determinable": 0.0 if target == "cannot_determine" else 1.0,
                },
            )
        )
    return items


def generate_negation_inactive_status_samples(count=300):
    items = []
    scenarios = [
        (
            "Meeting room C: free all afternoon, no bookings.",
            "Is meeting room C booked this afternoon? Answer strictly from the facts stated in the text.",
            False,
        ),
        (
            "Account settings: two-factor authentication is disabled.",
            "Is two-factor authentication active on this account?",
            False,
        ),
        (
            "Conference room A: fully booked until 5pm.",
            "Is conference room A currently available for walk-in use?",
            False,
        ),
        (
            "Security audit: biometric login is enabled for all administrators.",
            "Is biometric login active for administrators?",
            True,
        ),
        ("Subscription status: plan cancelled effective yesterday.", "Is the subscription currently active?", False),
        ("Facility status: gym is open to all hotel guests.", "Can hotel guests access the gym?", True),
        (
            "Server status: primary cluster is offline for scheduled maintenance.",
            "Is the primary cluster currently operational?",
            False,
        ),
        ("Device status: Bluetooth is powered on and discoverable.", "Is Bluetooth enabled on this device?", True),
    ]
    for _ in range(count):
        st, q_instr, expected_bool = random.choice(scenarios)
        items.append(
            bundle_multi_task_items(
                state=st,
                questions={
                    "is_active_or_booked": NoulQuestion(instructions=q_instr),
                },
                targets={
                    "is_active_or_booked": 1.0 if expected_bool else 0.0,
                },
            )
        )
    return items


def generate_hard_policy_settlement_samples(count=400):
    items = []
    scenarios = [
        (
            "Homeowners Policy HM-3: Water seepage covered up to $15,000 sublimit under Endorsement W-1. Damage estimate $22,400.",
            "pay_subject_to_15000_sublimit",
            ["pay_subject_to_15000_sublimit", "pay_full_estimate_less_deductible", "deny_vacancy_exclusion"],
        ),
        (
            "Corporate T&E Policy: Travel expenses require director sign-off under $50,000; CFO sign-off for commitments over $50,000. PR total $110,000.",
            "cfo",
            ["department_head", "director", "cfo", "budget_holder"],
        ),
        (
            "Export Control CCL: Advanced GPUs require validated export license for Group D destinations. Consignee in Group D.",
            "license_required",
            ["license_required", "license_exception_lvs", "no_license_required_nlr"],
        ),
        (
            "Conference Attendance Reimbursement: Lodging reimbursed up to EUR 400 with itemized invoice. Claim EUR 450 with receipt.",
            "eur_400",
            ["eur_300", "eur_400", "eur_450", "deny_unauthorized"],
        ),
        (
            "Warranty Claim Procedure: Unauthorized third-party repairs void entire device warranty. Client repaired screen at unauthorized shop.",
            "reject_entire_claim",
            ["approve_lodging_only", "reject_entire_claim", "pay_parts_only"],
        ),
        (
            "Sanctions Compliance Check: Export transaction lacks required end-user undertaking certificate.",
            "deny_current_export",
            ["deny_current_export", "privacy_only", "permit_with_monitoring"],
        ),
    ]
    for _ in range(count):
        state_text, target, options = random.choice(scenarios)
        criteria = {opt: f"Determination: {opt.replace('_', ' ').title()}" for opt in options}
        items.append(
            bundle_multi_task_items(
                state={"document": state_text},
                questions={
                    "ruling": ChoiceQuestion(
                        instructions="Apply the operative policy and select the required settlement determination:",
                        criteria=criteria,
                    ),
                    "is_rejected_or_capped": NoulQuestion(
                        instructions="Is the request rejected or capped by policy limits?"
                    ),
                },
                targets={
                    "ruling": target,
                    "is_rejected_or_capped": 1.0
                    if target in ["cfo", "license_required", "reject_entire_claim", "deny_current_export"]
                    or "sublimit" in target
                    else 0.0,
                },
            )
        )
    return items


def main():
    all_items = []
    all_items.extend(generate_trade_compliance_samples(400))
    all_items.extend(generate_policy_permission_samples(400))
    all_items.extend(generate_request_adequacy_samples(400))
    all_items.extend(generate_ticket_lifecycle_samples(400))
    all_items.extend(generate_alert_timeline_samples(400))
    all_items.extend(generate_concise_exception_policy_samples(400))
    all_items.extend(generate_format_adequacy_samples(400))
    all_items.extend(generate_ambiguity_insufficient_samples(300))
    all_items.extend(generate_negation_inactive_status_samples(300))
    all_items.extend(generate_hard_policy_settlement_samples(400))

    random.shuffle(all_items)
    print(f"[*] Generated {len(all_items)} targeted enterprise curriculum samples.")
    ClassOneDataset(all_items).save_jsonl(OUTPUT_PATH)
    print(f"[✓] Successfully saved to {OUTPUT_PATH}!")


if __name__ == "__main__":
    main()
