# 🔐 Security Policy

Two things live in this file: how to report a security issue **with this repository**, and the
security rules that govern all content **inside** it.

---

## Reporting a vulnerability

If you find a security problem in this repository — a committed secret, a malicious dependency,
an unsafe code example that could harm a learner — please report it privately:

1. Use GitHub's **Report a vulnerability** button under the Security tab, or
2. Open a confidential issue.

Please do **not** open a public issue for a live credential leak. Report it privately so it can be
revoked first.

Include: what you found, where, and why it is dangerous. We will acknowledge and respond as
promptly as we can.

---

## Never committed to this repository

- API keys, tokens, passwords
- Cloud credentials, SSH keys, certificates
- Private or personal datasets
- Personal data of any kind
- Proprietary model files

If a secret is committed by accident: **rotate it first**, then remove it from history. Rotation is
the fix; history rewriting is cleanup. A secret that has touched a public repository is burned.

Configuration is supplied through environment variables, documented in
[`.env.example`](.env.example). `.env` is git-ignored and must stay that way.

---

## Security rules for content

Every module that touches user input, retrieval, tool execution, model files or deployment carries
a **🔐 Security Note**. Specifically:

| Area | Required coverage |
| --- | --- |
| Anything accepting user text | Prompt injection, input validation |
| Anything retrieving documents | Indirect prompt injection, retrieval poisoning |
| Anything calling tools | Tool permissions, sandboxing, approval gates, cost limits |
| Anything loading models | Unsafe deserialisation, provenance of weights |
| Anything deployed | Authn/authz, rate limiting, tenant isolation, audit logging |
| Anything multi-tenant | Data isolation, per-tenant quotas, leakage between tenants |

### Unsafe patterns we always flag

<!-- check-examples: skip -->
```python
# ❌ Arbitrary code execution during deserialisation
model = pickle.load(open(untrusted_path, "rb"))

# ❌ Executing model output
result = eval(llm_response)

# ❌ Unvalidated tool arguments from a model
subprocess.run(llm_response["command"], shell=True)

# ❌ Rendering model output as raw HTML
return f"<div>{llm_response}</div>"

# ❌ Secret in source
api_key = "sk-..."
```

Each appears in the content **only** alongside its safe replacement.

---

## Defensive only

All security material here is defensive. This repository does not provide offensive tooling,
detection-evasion techniques, working exploit code, or instructions for attacking systems you do
not own. Red-teaming content teaches you to test **your own** systems, within an authorised scope.

---

## AI-specific threat checklist

Used by [`28-ai-security/`](28-ai-security/README.md) and by every production project:

- [ ] Direct prompt injection considered and mitigated
- [ ] Indirect prompt injection via retrieved content considered
- [ ] Model output treated as untrusted data everywhere it is used
- [ ] Tool calls have an allowlist, schema validation and permission boundaries
- [ ] Tool execution is sandboxed; destructive actions require human approval
- [ ] Rate limits and cost caps enforced per user and per tenant
- [ ] Sensitive data is not sent to third-party model providers without a documented decision
- [ ] Training and retrieval corpora are checked for poisoning and for personal data
- [ ] Model files come from a verified source; no `pickle` on untrusted input
- [ ] Dependencies pinned, scanned and reviewed; containers scanned
- [ ] Audit logs capture prompts, tool calls and decisions for after-the-fact review
- [ ] Tenants are isolated at the data, index and cache layers
- [ ] A rollback path exists for a bad model or prompt version

---

## Authoritative guidance

Consult current versions of established AI-security guidance when building the security module or
any production project. Vendor and standards-body documents change; always check the current
edition and record the date you verified it.

---

## Dependencies

- All dependencies are pinned.
- `pip-audit` runs in CI.
- New dependencies require justification in the pull request.
- Prefer the standard library where it is genuinely sufficient.

---

[🏠 Repository Home](README.md) · [AI Security module →](28-ai-security/README.md)
