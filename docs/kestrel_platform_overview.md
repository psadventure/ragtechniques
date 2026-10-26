# Kestrel Core Banking Platform — Product Overview

> **Fictional sample document.** Kestrel, Kestrel Systems, Harbor & Pine Credit Union and every name, number and date in this file are invented for RAG experiments. Any resemblance to real products or organisations is coincidental.

Kestrel is a cloud-native core banking platform built by Kestrel Systems for credit unions and community banks. This overview describes Kestrel 4.2: what it does, how it is built, and how it is deployed.

## 1. What Kestrel Is

Kestrel replaces the "system of record" at a financial institution: the software that holds member accounts, posts every transaction to the general ledger, moves money between institutions, and decides who may open an account or receive a loan. Traditional cores run nightly batch jobs on a mainframe; Kestrel processes every event in real time and exposes all capabilities through a versioned REST API.

Kestrel is organised into six services that can be licensed together or separately:

| Service | Responsibility | Typical owner at the bank |
|---|---|---|
| `kestrel-ledger` | Accounts, balances, double-entry journal, interest | Finance / Operations |
| `kestrel-payments` | Book transfers, ACH, wires, instant payments | Payments Operations |
| `fraud-shield` | Real-time fraud scoring and case management | Risk / Fraud team |
| `kestrel-lending` | Loan origination, decisioning and servicing | Lending |
| `kyc-gate` | Digital onboarding and identity verification | Member Services / Compliance |
| `kestrel-edge` | API gateway, authentication, rate limiting | IT / Digital |

The largest current implementation is at **Harbor & Pine Credit Union**, which serves 180,000 members across 42 branches. Harbor & Pine is scheduled to go live on Kestrel in **September 2026**, replacing its legacy CoreOne system (see the Project Heron plan for the detailed timeline).

## 2. Architecture

Every Kestrel service owns its own database and communicates through an event stream. A transfer request, for example, flows through the gateway, is scored by Fraud Shield, executed by the payments service and finally posted by the ledger. Each step publishes an event that downstream services and the bank's own systems can subscribe to.

```text
  Mobile / Web / Branch apps          Bank back-office systems
            │                                   ▲
            ▼                                   │ webhooks
   ┌────────────────┐   OAuth 2.0   ┌───────────┴──────────┐
   │  kestrel-edge  │──────────────►│   event stream (Kafka) │
   └───────┬────────┘               └──▲────────▲────────▲──┘
           │ REST                       │        │        │
   ┌───────▼────────┐  score   ┌────────┴──┐ ┌───┴──────┐ ┌┴───────────────┐
   │kestrel-payments│─────────►│fraud-shield│ │kyc-gate  │ │kestrel-lending │
   └───────┬────────┘          └───────────┘ └──────────┘ └────────────────┘
           │ post
   ┌───────▼────────┐
   │ kestrel-ledger │  immutable double-entry journal
   └────────────────┘
```

Key architectural properties:

- **Immutable journal.** Ledger entries are never updated or deleted. Corrections are made with reversing entries, which gives auditors a complete history.
- **Idempotent writes.** Every money-moving API call requires an idempotency key, so retries after a network failure can never post a payment twice.
- **Event sourcing.** Account balances are projections of the journal and can be rebuilt at any time.
- **Multi-region.** Production runs active-passive across two regions with a recovery point objective (RPO) of 5 minutes and a recovery time objective (RTO) of 1 hour.

## 3. Core Features

This section describes the functional capabilities of each Kestrel service in detail. Configuration values mentioned here are defaults; each institution can change them within the limits set by its regulator.

### 3.1 Accounts and Ledger

The ledger supports savings, checking (share draft), money market, certificates of deposit and loan accounts. Every account belongs to one or more members and maps to a general ledger (GL) account for financial reporting. Kestrel uses strict double-entry bookkeeping: each transaction creates at least two journal lines whose debits and credits sum to zero.

Interest accrues daily on the closing balance and is posted monthly on the last business day. The **end-of-day close** runs at 23:30 Eastern Time. During the close, Kestrel freezes the business date, accrues interest, applies fees, and produces the daily trial balance. Transactions received during the close are queued and posted with the next business date; members still see them immediately as pending.

Overdraft protection can sweep funds from a linked savings account. Sweeps happen in real time, not at end of day, which removes most overnight overdraft fees.

### 3.2 Payments

Kestrel supports four payment rails:

- **Book transfers** between accounts at the same institution settle instantly, around the clock.
- **ACH** payments are sent in same-day ACH windows with submission cutoffs at 10:30 and 14:45 Eastern Time. Items submitted after 14:45 go out in the next-day batch.
- **Wires** are sent through Fedwire with a cutoff of 17:00 Eastern Time on business days.
- **Instant payments** (RTP and FedNow) settle in seconds, 24/7, up to **$25,000 per transaction** by default.

Default daily outgoing limits are **$10,000 for retail members** and **$250,000 for business members**. Limits apply across all rails combined and reset at midnight Eastern Time. A member can request a temporary limit increase, which a staff member approves in the back-office console.

### 3.3 Fraud Shield

Fraud Shield scores every outgoing payment and every login in real time and returns a **risk score from 0 to 1000**. The p95 scoring latency is 45 milliseconds, so scoring never noticeably slows a payment. The score combines a machine-learning model trained on the institution's history with a set of explainable rules (see section 4).

What happens next depends on the score: low-risk activity proceeds, medium-risk activity requires the member to confirm with a one-time passcode (a "step-up"), and high-risk activity is blocked and opens a case for the fraud team. Analysts work cases in the Fraud Shield console, where they see the triggered rules, the member's recent activity and a timeline of related logins.

### 3.4 Lending

The lending service originates personal loans, auto loans and home-equity lines of credit (HELOCs). Applications arrive through the API or the branch console. The decision engine **auto-approves applications with a credit score of 700 or higher and a debt-to-income (DTI) ratio of 36% or lower**, provided the requested amount is within the product's limits. All other applications are routed to an underwriter queue for manual review. Approved loans are boarded onto the ledger automatically, with the repayment schedule generated at boarding.

### 3.5 Member Onboarding (kyc-gate)

kyc-gate lets a new member open an account from a phone in a single session. It checks a government ID document, performs a selfie liveness check, screens the applicant against OFAC sanctions lists and verifies the residential address. The median onboarding time is 6 minutes. Kestrel targets **85% straight-through processing**, meaning 85% of applicants are approved without any staff involvement. Applications that fail a check go to a review queue with the specific failing check highlighted.

## 4. Fraud Shield Scoring Rules

Rules add points to the model score. The final score determines the action:

| Score | Action | Member experience |
|---|---|---|
| 0 – 599 | Allow | Payment proceeds immediately |
| 600 – 849 | Step-up | Member must confirm with a one-time passcode |
| 850 – 1000 | Block | Payment rejected, fraud case opened |

The built-in rules are:

| Rule ID | Name | Triggers when | Points |
|---|---|---|---|
| FS-101 | Velocity | More than 5 outgoing transfers within 10 minutes | +250 |
| FS-204 | New payee, large amount | First payment to a payee and amount above $2,500 | +300 |
| FS-310 | Geo mismatch | Login country differs from the card's last-used country within 30 minutes | +200 |
| FS-415 | Dormant reactivation | Account had no activity for 180 days, then sends money | +350 |
| FS-520 | Mule pattern | More than 90% of an inbound deposit is sent out within 15 minutes | +400 |

Institutions tune thresholds and rule weights in a YAML policy file. Changes are versioned and take effect without a deployment:

```yaml
# fraud-shield policy — Harbor & Pine Credit Union (excerpt)
policy_version: 2026.07
thresholds:
  step_up: 600
  block: 850
rules:
  FS-101:            # velocity
    enabled: true
    window_minutes: 10
    max_transfers: 5
    points: 250
  FS-204:            # new payee, large amount
    enabled: true
    amount_over: 2500
    points: 300
  FS-415:            # dormant reactivation
    enabled: true
    dormant_days: 180
    points: 350
  FS-520:            # mule pattern
    enabled: true
    outbound_ratio: 0.90
    window_minutes: 15
    points: 400
allow_lists:
  payees: ["payroll-providers", "irs-direct-pay"]
```

## 5. Deployment Options

| Option | Tenancy | Who operates it | Typical customer |
|---|---|---|---|
| Kestrel Cloud | Multi-tenant SaaS | Kestrel Systems | Institutions under $1B in assets |
| Kestrel Dedicated | Single-tenant, private network in the vendor's cloud account | Kestrel Systems | Institutions with data-isolation requirements |
| Kestrel On-Prem | Customer's own Kubernetes cluster (version 1.29 or later, at least 12 nodes) | The institution, with vendor support | Institutions with regulatory constraints on cloud use |

Harbor & Pine Credit Union runs **Kestrel Dedicated** in the US East region with its disaster-recovery site in US Central. Upgrades are delivered quarterly; Dedicated and On-Prem customers can defer an upgrade by at most one quarter.

## 6. Security and Compliance

Kestrel Systems holds a SOC 2 Type II report, renewed annually. Services that handle card data are certified against PCI DSS 4.0. All data is encrypted at rest with AES-256 and in transit with TLS 1.3. Encryption keys are held in a hardware security module, and Dedicated customers can bring their own keys.

Staff access to production requires single sign-on with phishing-resistant multi-factor authentication and is granted just-in-time for a maximum of 8 hours. Every staff action in the back-office console is written to a tamper-evident audit log retained for 7 years.

## 7. Glossary

| Term | Meaning |
|---|---|
| Core | The system of record for accounts and the general ledger |
| GL | General ledger: the institution's master accounting record |
| Book transfer | A transfer between two accounts at the same institution |
| Step-up | An extra verification step (e.g. one-time passcode) required for a risky action |
| Straight-through processing | Completing a process with no manual intervention |
| Idempotency key | A unique value sent with a request so that retries are not executed twice |
| RPO / RTO | Recovery point objective (max data loss) / recovery time objective (max downtime) |
