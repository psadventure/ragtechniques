# Project Heron — Kestrel Rollout at Harbor & Pine Credit Union

> **Fictional sample document.** Project Heron, Harbor & Pine Credit Union, Kestrel, CoreOne and every person, date and amount in this file are invented for RAG experiments.

Project Heron is the program that moves Harbor & Pine Credit Union from its legacy core system, CoreOne, to the Kestrel Core Banking Platform. This document is the program plan of record. It is updated after every steering committee; the most recent revision is listed in the change history at the end.

## 1. Project Summary

| Item | Value |
|---|---|
| Sponsor | Harbor & Pine Credit Union executive committee |
| Scope | Ledger, payments, Fraud Shield, lending, digital onboarding, API |
| Members affected | 180,000 members, 42 branches |
| Budget | $4.8 million, plus 12% contingency |
| Kickoff | 12 January 2026 |
| Go-live (cutover weekend) | 17–18 October 2026 |
| CoreOne decommission | 29 January 2027 |
| Deployment model | Kestrel Dedicated, US East (DR in US Central) |

The program's goals are to retire CoreOne before its support contract ends in March 2027, to give members real-time payments and digital onboarding, and to reduce fraud losses with Fraud Shield.

## 2. Team and Ownership

| Role | Name | Organisation |
|---|---|---|
| Program director | Maya Okafor | Harbor & Pine |
| Delivery lead | Tomás Reyes | Kestrel Systems |
| Data migration lead | Priya Raman | Harbor & Pine |
| QA and test lead | Daniel Brooks | Harbor & Pine |
| Information security officer | Lena Fischer | Harbor & Pine |
| Fraud Shield implementation | Arjun Mehta | Kestrel Systems |

The steering committee meets every second Thursday and owns all changes to scope, budget and go-live dates.

## 3. Timeline at a Glance

| Phase | Name | Start | End | Status |
|---|---|---|---|---|
| 0 | Discovery and design | 2026-01-12 | 2026-02-20 | Complete |
| 1 | Ledger and accounts migration | 2026-02-23 | 2026-05-29 | Complete |
| 2 | Payments and API | 2026-04-06 | 2026-07-31 | Complete |
| 3 | Fraud Shield pilot and rollout | 2026-04-13 | 2026-08-17 | Complete |
| 4 | Lending and onboarding | 2026-06-01 | 2026-09-25 | In progress |
| 5 | Cutover and go-live | 2026-10-17 | 2026-10-18 | Planned |
| 6 | Hypercare | 2026-10-19 | 2026-11-27 | Planned |
| 7 | CoreOne decommission | 2026-12-01 | 2027-01-29 | Planned |

## 4. Phase Details

Each phase has an owner, a set of deliverables and exit criteria that the steering committee must sign off before the next dependent phase starts.

### Phase 0 — Discovery and design

Owner: Maya Okafor. The team mapped 1,140 CoreOne product codes to 212 Kestrel product configurations, documented 38 interfaces to other systems, and agreed the target general ledger structure with Finance. The main output was the solution design document, signed off on 20 February 2026. Discovery found that 9% of CoreOne member records had missing or invalid addresses, which became risk R1.

### Phase 1 — Ledger and accounts migration

Owner: Priya Raman. Migration is rehearsed with **mock conversions**: a full copy of CoreOne data is converted and loaded into a Kestrel test environment, then reconciled. Three mock conversions were held on 20 March, 24 April and 15 May 2026. The first mock had a reconciliation variance of $41,207.13 caused by rounding in dividend accruals; the second reduced it to $312.50; the third reconciled to **$0.00 across all GL accounts**, meeting the exit criterion. Address clean-up brought invalid addresses down from 9% to 0.4%.

### Phase 2 — Payments and API

Owner: Tomás Reyes. Phase 2 connected Kestrel to the ACH operator, Fedwire, and the RTP and FedNow networks, and rebuilt the 14 internal integrations that previously used CoreOne's batch files on top of the Kestrel API and webhooks. Load testing at twice the expected peak (1,800 requests per second) produced a p95 API latency of 210 ms, inside the 300 ms target.

### Phase 3 — Fraud Shield pilot and rollout

Owner: Arjun Mehta. The pilot was originally planned to start on 2 March 2026 but **started on 13 April 2026** (decision D-07) because the export of three years of CoreOne fraud history, needed to train the scoring model, was delayed. The pilot covered 5% of members for eight weeks. The false-positive rate fell from 3.1% in the first week to 1.6% by the end of the pilot, below the 2.0% target, after the FS-204 threshold was raised from $1,000 to $2,500. Fraud Shield reached general availability for all members on 17 August 2026.

### Phase 4 — Lending and onboarding

Owner: Daniel Brooks. This phase configures kestrel-lending for personal, auto and HELOC products and launches kyc-gate digital onboarding. Underwriters asked for the auto-approval rule to also require at least 12 months of membership for HELOCs, which Kestrel supports as a product-level setting. User acceptance testing runs until 25 September 2026.

## 5. Milestones and Go/No-Go Criteria

The steering committee holds a go/no-go meeting one week before cutover. All criteria must be met:

| # | Criterion | Target | Latest result |
|---|---|---|---|
| G1 | Final mock conversion reconciliation | $0.00 variance in every GL account | $0.00 (mock 3) |
| G2 | Automated regression suite | ≥ 99.5% passing | 99.7% |
| G3 | Fraud Shield false-positive rate | ≤ 2.0% | 1.6% |
| G4 | API latency at 2× peak load | p95 ≤ 300 ms | 210 ms |
| G5 | Disaster-recovery failover test | Recover within RTO of 1 hour | Scheduled 2026-10-03 |
| G6 | PCI DSS remediation | All audit findings closed | Closed 2026-09-18 |
| G7 | Staff training | ≥ 95% of branch staff trained | 91% (on track) |

## 6. Risk Register

| ID | Risk | Likelihood | Impact | Mitigation | Owner |
|---|---|---|---|---|---|
| R1 | Poor legacy data quality (addresses, duplicates) | High | High | Clean-up sprints before each mock conversion | Priya Raman |
| R2 | Key-person dependency on the CoreOne specialist | Medium | High | Pairing and recorded walkthroughs of CoreOne batch jobs | Maya Okafor |
| R3 | PCI audit findings delay go-live | Medium | High | Dedicated remediation squad; weekly status to steering | Lena Fischer |
| R4 | Vendor delivery capacity during other Kestrel rollouts | Low | Medium | Named Kestrel team contractually reserved until hypercare ends | Tomás Reyes |
| R5 | Member confusion at cutover (new cards, app re-login) | Medium | Medium | Four-wave member communications starting 8 weeks before go-live | Maya Okafor |

## 7. Decision Log

| ID | Date | Decision | Rationale |
|---|---|---|---|
| D-01 | 2026-01-15 | Deploy on Kestrel Dedicated, not Kestrel Cloud | Board requires single-tenant data isolation |
| D-03 | 2026-02-12 | Migrate 7 years of transaction history, archive the rest | Balances statement needs vs. conversion time |
| D-07 | 2026-02-26 | Move Fraud Shield pilot start from 2 March to 13 April 2026 | CoreOne fraud-history export delayed by 6 weeks |
| D-09 | 2026-05-21 | Raise FS-204 new-payee threshold from $1,000 to $2,500 | Pilot false-positive rate too high at $1,000 |
| D-11 | 2026-07-09 | Move go-live from 12–13 September to 17–18 October 2026 | PCI DSS audit found 4 findings needing remediation before cutover |
| D-12 | 2026-07-23 | Freeze CoreOne product changes from 1 September 2026 | Avoid re-mapping during final conversion |

## 8. Cutover Plan

Cutover runs over the weekend of 17–18 October 2026. CoreOne stops accepting transactions at 18:00 Eastern Time on Friday 16 October; branches close as usual on Saturday. The final conversion starts at 19:00 on Friday, reconciliation is signed off by Finance by 14:00 on Saturday, and Kestrel opens to members at 06:00 on Sunday 18 October. If reconciliation fails, the rollback decision point is **Saturday 20:00**: CoreOne is reopened and the go-live moves to the next available weekend.

## 9. Change History

| Revision | Date | Change |
|---|---|---|
| 1.0 | 2026-02-20 | Baseline plan after discovery; go-live 12–13 September 2026 |
| 1.3 | 2026-02-26 | Fraud Shield pilot moved to 13 April (D-07) |
| 1.6 | 2026-07-09 | Go-live moved to 17–18 October 2026 (D-11) |
| 1.8 | 2026-09-24 | Updated go/no-go results; PCI remediation closed |
