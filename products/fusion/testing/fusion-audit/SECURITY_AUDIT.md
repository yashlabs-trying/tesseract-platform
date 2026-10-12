# Fusion — Security Audit

**Scope:** Tesseract-Fusion API, settings, allocator/optimizer, and Kubernetes RBAC
**Posture:** Advisory-only by default; read-only cluster access; no mutation endpoints.
**Result:** No Critical/High open findings. Two Medium findings identified and **fixed**
(`api_key` repr leak; unenforced `resource_quotas`). Remaining items are **documented limitations**.

> All findings below are grounded in the stated evidence. Where a control is not implemented, it is
> recorded as a documented limitation rather than claimed as mitigated.

---

## 1. Findings summary

| ID | Severity | Component | Finding | Evidence | Status |
|----|----------|-----------|---------|----------|--------|
| SEC-01 | Medium | Settings / `api_key` | API key leaked via `repr` | `api_key` appeared in object `repr` | **Fixed** — now `SecretStr` |
| SEC-02 | Medium | Multi-tenancy / quotas | `resource_quotas` not enforced | Quotas accepted but ignored | **Fixed** — quotas now enforced |
| SEC-03 | Low | Authentication | No authentication enabled by default | `enable_auth=False` | Open — documented limitation |
| SEC-04 | Low | API availability | No rate limiting on expensive optimization jobs | No limiter on optimize endpoints | Open — documented limitation |
| SEC-05 | Low | Input validation | No input size limit enforcement | No enforced size cap on request payloads | Open — documented limitation |
| SEC-06 | Info | API behavior | Malformed content returned 500 (info disclosure / poor UX) | 500 on malformed content | **Fixed** — now 422 |
| SEC-07 | Info | Allocation integrity | Unhealthy GPUs silently filtered | Silently dropped instead of surfaced | **Fixed** — explicit rejection |
| SEC-08 | Info | Validation | Pydantic accepted negatives | Negative values accepted | **Fixed** — `ge=0` |
| SEC-09 | Info | Execution | No arbitrary code execution | No `eval`/`exec` in optimizer | Verified — no finding |

---

## 2. Fixed findings (detail)

### SEC-01 — `api_key` leaked in `repr` (Medium, FIXED)

- **Affected component:** Settings object.
- **Evidence:** Before the fix, the API key was rendered in the object's `repr`, making it easy to
  leak into logs, tracebacks, or debug output.
- **Remediation:** The field is now a `SecretStr`, so the value is masked in `repr` and similar
  representations.
- **Status:** Fixed.

### SEC-02 — `resource_quotas` not enforced (Medium, FIXED)

- **Affected component:** Multi-tenancy / quota enforcement.
- **Evidence:** `resource_quotas` were accepted by configuration but not applied to allocation
  decisions.
- **Remediation:** Quotas are now enforced during allocation. Combined with per-tenant policies and
  `max_concurrent_workers`, tenant isolation is respected.
- **Status:** Fixed.

### SEC-06 — API 500 on malformed content (Info, FIXED)

- **Affected component:** API error handling.
- **Evidence:** Malformed request content produced HTTP 500.
- **Remediation:** Now returns HTTP 422 (Unprocessable Entity), which is the correct semantic for
  invalid input and avoids implying a server fault.
- **Status:** Fixed.

### SEC-07 — Unhealthy GPUs silently filtered (Info, FIXED)

- **Affected component:** Allocator GPU selection.
- **Evidence:** Unhealthy GPUs were silently dropped from consideration.
- **Remediation:** Unhealthy GPUs are now explicitly rejected/surfaced rather than silently removed.
- **Status:** Fixed.

### SEC-08 — Pydantic accepted negatives (Info, FIXED)

- **Affected component:** Input models.
- **Evidence:** Negative values (e.g., negative GPU memory) were accepted by validation.
- **Remediation:** Constraints added (`ge=0`), so negative values are rejected.
- **Status:** Fixed.

---

## 3. Open limitations (documented)

| ID | Limitation | Impact | Rationale / note |
|----|------------|--------|------------------|
| SEC-03 | No authentication by default (`enable_auth=False`) | Anyone who can reach the API can call it | Acceptable only for trusted/local deployments; must be enabled for any shared or network-exposed deployment |
| SEC-04 | No rate limiting on expensive optimization jobs | Potential resource-exhaustion / DoS via repeated heavy jobs | Optimization is CPU-heavy; a limiter is recommended before exposing beyond trusted users |
| SEC-05 | No input size limit enforcement | Very large payloads could consume memory/CPU | A request-size cap is recommended |

These are **not** claimed as mitigated. They are explicit, documented limitations.

---

## 4. Positive controls (verified)

### 4.1 Advisory-only by default

- `Settings.advisory_mode=True`
- `execution_eligibility=False`

Fusion advises; it does not execute or mutate cluster state by default.

### 4.2 No mutation endpoints

The API exposes only:

- `POST /import`
- `POST /evaluate`
- `POST /simulate`
- `GET` endpoints

There are **no `PUT`, `PATCH`, or `DELETE`** endpoints. There is no path to mutate cluster or
scheduling state through the API.

### 4.3 Kubernetes RBAC is read-only

| Aspect | Value |
|--------|-------|
| Kind | `ClusterRole` |
| Resources | nodes, pods, namespaces, deployments, statefulsets |
| Verbs | `get`, `list`, `watch` |
| Write verbs (`create`, `update`, `patch`, `delete`) | **Absent** |

Fusion cannot mutate the cluster via its Kubernetes credentials.

### 4.4 No arbitrary code execution

No `eval`/`exec` (or equivalent dynamic execution) exists in the optimizer. Optimization is bounded
to the solver paths (exhaustive / CP-SAT / greedy).

### 4.5 Tenant isolation

- Per-tenant policies are respected.
- `resource_quotas` enforced (post-fix).
- `max_concurrent_workers` enforced.

### 4.6 Secrets handling

- `api_key` is now `SecretStr` (no `repr` leak).

---

## 5. Abuse-case tests

| Abuse case | Injected how | Observed behavior | Outcome |
|------------|--------------|-------------------|---------|
| Malformed specs | Send malformed content to API | Returns 422 (was 500 pre-fix) | Handled — no server fault |
| Negative resource values | Submit negative GPU memory / resources | Rejected by validation (`ge=0`) | Handled |
| Quota manipulation | Configure/attempt to bypass `resource_quotas` | Quotas enforced (post-fix); cannot exceed | Handled |
| Priority abuse | Submit workloads with abusive priority settings | Bounded by per-tenant policies; no privilege escalation path | Handled within tenant policy |
| Input flooding | Repeated expensive optimization calls | No rate limit (SEC-04) | **Documented limitation** |
| Oversized payloads | Very large request body | No enforced size cap (SEC-05) | **Documented limitation** |

---

## 6. Caveats

- No authentication and no rate limiting means the API is suitable only for trusted/controlled
  environments in its default configuration.
- Kubernetes validation was performed on a CPU-only kind cluster with read-only RBAC; no real GPU
  hardware security testing was performed.
- This audit covers the Fusion service; it does not certify the host cluster, the container runtime,
  or third-party dependencies beyond what was exercised.
