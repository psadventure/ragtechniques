# Kestrel API Reference (v2)

> **Fictional sample document.** Kestrel and every endpoint, error code and value in this file are invented for RAG experiments. Hostnames use the reserved `.example` domain and do not resolve.

The Kestrel API gives programmatic access to accounts, transfers and events on the Kestrel Core Banking Platform. It is a JSON-over-HTTPS REST API served by `kestrel-edge`, the platform's API gateway.

## 1. Overview

| Environment | Base URL | Data |
|---|---|---|
| Sandbox | `https://sandbox.api.kestrel.example/v2` | Synthetic accounts, no real money |
| Production | `https://api.kestrel.example/v2` | Live member data |

General conventions:

- All requests and responses use JSON with UTF-8 encoding.
- Amounts are objects with a decimal string `value` and an ISO 4217 `currency`, for example `{"value": "125.00", "currency": "USD"}`. Never send amounts as floating-point numbers.
- Timestamps are ISO 8601 in UTC, for example `2026-05-04T14:30:00Z`.
- List endpoints are paginated with an opaque `cursor`. The maximum page size is 200 items.
- Every response includes a `Request-Id` header. Include it when contacting support.

## 2. Authentication

Kestrel uses the OAuth 2.0 client-credentials flow. Each integration is registered in the back-office console and receives a client ID and client secret. Exchange them for an access token at the token endpoint:

```bash
curl -X POST https://sandbox.api.kestrel.example/oauth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d grant_type=client_credentials \
  -d client_id=$KESTREL_CLIENT_ID \
  -d client_secret=$KESTREL_CLIENT_SECRET \
  -d scope="accounts:read transfers:write"
```

Access tokens are valid for **3600 seconds** (one hour). Request a new token before expiry; refresh tokens are not issued for the client-credentials flow. Send the token in the `Authorization: Bearer <token>` header on every call.

| Scope | Grants |
|---|---|
| `accounts:read` | Read accounts, balances and transactions |
| `transfers:write` | Create and cancel transfers |
| `transfers:read` | Read transfer status |
| `webhooks:manage` | Create, update and delete webhook subscriptions |

## 3. Accounts API

| Method | Path | Description |
|---|---|---|
| GET | `/accounts` | List accounts visible to the client |
| GET | `/accounts/{account_id}` | Get one account |
| GET | `/accounts/{account_id}/balances` | Current, available and pending balances |
| GET | `/accounts/{account_id}/transactions` | Posted transactions, filter with `from`, `to`, `cursor` |

The **available balance** is the current balance minus holds and pending debits; it is the amount a member can spend right now. Example balance response:

```json
{
  "account_id": "acc_7Hq2Lm9Pz",
  "currency": "USD",
  "current": {"value": "4210.55", "currency": "USD"},
  "available": {"value": "3960.55", "currency": "USD"},
  "pending_debits": {"value": "250.00", "currency": "USD"},
  "as_of": "2026-05-04T14:30:00Z"
}
```

## 4. Transfers API

Transfers move money out of a member's account over one of four rails. Creating a transfer is asynchronous: the API returns immediately with a status, and the final outcome arrives by webhook or by polling.

### 4.1 Create a transfer

`POST /transfers` requires the `transfers:write` scope and an **`Idempotency-Key` header** containing a client-generated UUID. Kestrel stores the key and the request body for **24 hours**. If you retry with the same key and the same body, you get the original response and no new transfer is created. If you reuse a key with a *different* body, the request fails with error `KST-4090`.

Request body fields:

| Field | Required | Notes |
|---|---|---|
| `source_account_id` | yes | Account to debit |
| `destination` | yes | `{"account_id": ...}` for book transfers, or `{"routing_number": ..., "account_number": ...}` for external |
| `amount` | yes | Amount object; must be greater than zero |
| `rail` | yes | One of `book`, `ach`, `wire`, `instant` |
| `memo` | no | Up to 140 characters, shown on the member's statement |
| `scheduled_for` | no | Future date for ACH and wire transfers |

```python
import os
import uuid
from kestrel import KestrelClient

client = KestrelClient(
    client_id="cli_live_8d2f",
    client_secret=os.environ["KESTREL_CLIENT_SECRET"],
    environment="sandbox",
)

transfer = client.transfers.create(
    idempotency_key=str(uuid.uuid4()),   # store this with your own record before calling
    source_account_id="acc_7Hq2Lm9Pz",
    destination={"routing_number": "021000021", "account_number": "000123456789"},
    amount={"value": "1500.00", "currency": "USD"},
    rail="ach",
    memo="Invoice 2026-118",
)
print(transfer.id, transfer.status)      # e.g. trf_3Kd9Q pending
```

### 4.2 Transfer lifecycle

| Status | Meaning |
|---|---|
| `pending` | Accepted, not yet submitted to the rail; can still be cancelled |
| `processing` | Submitted to the rail |
| `held_for_review` | Fraud Shield required a step-up or analyst review |
| `completed` | Settled successfully |
| `failed` | Rejected before settlement; funds were never debited |
| `returned` | Settled, then returned by the receiving bank (ACH only); funds are credited back |

A transfer in `held_for_review` resumes automatically once the member completes the one-time-passcode step-up. If the member does not respond within 30 minutes, the transfer moves to `failed` with error `KST-4229`.

### 4.3 Retrieve and cancel

`GET /transfers/{transfer_id}` returns the current state, including the rail reference number once the transfer is processing. `POST /transfers/{transfer_id}/cancel` cancels a transfer, but **only while it is `pending`**. Cancelling a transfer in any other state returns `KST-4001`. Wires and instant payments move from `pending` to `processing` within seconds, so in practice only scheduled ACH and wire transfers can be cancelled.

## 5. Webhooks

Webhooks notify your system when something happens, so you don't need to poll. Subscriptions are managed through `/webhooks` with the `webhooks:manage` scope.

| Event | Sent when |
|---|---|
| `transfer.completed` | A transfer settled |
| `transfer.failed` | A transfer failed or was cancelled |
| `transfer.held_for_review` | Fraud Shield held a transfer |
| `account.opened` | kyc-gate approved and opened a new account |
| `fraud.alert` | Fraud Shield blocked an action and opened a case |

Every delivery carries a `Kestrel-Signature` header in the form `t=<unix timestamp>,v1=<hex signature>`. The signature is an HMAC-SHA256 of the string `"<timestamp>.<raw request body>"`, keyed with the subscription's signing secret. **Reject any delivery whose timestamp is more than 300 seconds old**; this blocks replay attacks.

```python
import hashlib, hmac, time

def verify_kestrel_signature(raw_body: bytes, header: str, secret: str, tolerance: int = 300) -> bool:
    parts = dict(item.split("=", 1) for item in header.split(","))
    timestamp, received = parts["t"], parts["v1"]
    if abs(time.time() - int(timestamp)) > tolerance:
        return False                                    # too old: possible replay
    signed_payload = f"{timestamp}.".encode() + raw_body
    expected = hmac.new(secret.encode(), signed_payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, received)      # constant-time comparison
```

Respond with any 2xx status within 10 seconds. Failed deliveries are retried with exponential backoff, up to **8 attempts over 24 hours**, after which the subscription is marked unhealthy in the console.

## 6. Rate Limits

| Tier | Requests per minute | Burst |
|---|---|---|
| Sandbox | 50 | 10 |
| Standard | 600 | 100 |
| Premium | 3000 | 500 |

Every response includes `X-RateLimit-Limit` and `X-RateLimit-Remaining` headers. When you exceed the limit you receive HTTP 429 with error `KST-4290` and a `Retry-After` header giving the number of seconds to wait. Limits are counted per client ID, not per IP address.

## 7. Error Codes

Errors return a JSON body with `code`, `message` and `request_id`.

| Code | HTTP | Name | What to do |
|---|---|---|---|
| KST-4001 | 400 | invalid_request | Fix the request; the message names the field |
| KST-4010 | 401 | invalid_token | Get a new access token |
| KST-4030 | 403 | insufficient_scope | Request the missing scope for your client |
| KST-4041 | 404 | account_not_found | Check the account ID and that your client may see it |
| KST-4090 | 409 | idempotency_conflict | Same Idempotency-Key reused with a different body; generate a new key |
| KST-4220 | 422 | insufficient_funds | Available balance is lower than the amount |
| KST-4221 | 422 | limit_exceeded | Daily or per-transaction limit exceeded |
| KST-4229 | 422 | fraud_blocked | Blocked by Fraud Shield or step-up not completed; do not retry automatically |
| KST-4290 | 429 | rate_limited | Wait for `Retry-After` seconds, then retry |
| KST-5000 | 500 | internal_error | Retry with the same Idempotency-Key |
| KST-5030 | 503 | rail_unavailable | The payment rail is closed or down; retry later or use another rail |

## 8. SDK Quickstart

The official Python SDK is published as `kestrel-sdk` and handles token refresh, retries with idempotency keys, pagination and webhook signature verification.

```bash
pip install kestrel-sdk
```

```python
from kestrel import KestrelClient

client = KestrelClient.from_env()          # reads KESTREL_CLIENT_ID / KESTREL_CLIENT_SECRET

for account in client.accounts.list():     # pagination handled for you
    balances = client.accounts.balances(account.id)
    print(account.id, account.type, balances.available.value)
```

The SDK retries `KST-5000` and `KST-5030` automatically up to 3 times with the same idempotency key. It never retries `KST-4229`, because repeating a blocked payment can escalate the fraud case.
