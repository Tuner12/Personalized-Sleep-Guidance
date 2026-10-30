"""One way to call an agent and get a validated object back.

Every reasoning step in this pipeline has the same shape: send a prompt plus a JSON
payload, get a structured object, reject it if it does not satisfy the contract. So
there is one client, and the roles differ only by prompt and schema. Validation is
part of the call rather than each caller's business, which is what lets the pipeline
read as four steps instead of four steps interleaved with parsing.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

ModelT = TypeVar("ModelT", bound=BaseModel)


class StructuredCallError(RuntimeError):
    """The agent failed, returned unparseable text, or broke the contract.

    One exception type for all three because callers treat them identically: the
    attempt failed, record why, and let the repair loop decide whether to retry.
    """


@dataclass(frozen=True)
class LLMClient:
    """A named Codex/Claude Code role with schema-validated output."""

    provider: str
    command: str
    args: tuple[str, ...] = ()
    protocol: str = "stdin_json"
    timeout_s: int = 600
    protocol_attempts: int = 2
    transient_attempts: int = 4
    transient_backoff_s: float = 10.0
    # Batch windows may all stay alive, while this machine-wide limit keeps the
    # number of active external Codex requests within the account allowance.
    max_concurrent_calls: int = 10
    concurrency_lock_dir: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def structured(
        self,
        *,
        prompt: str,
        payload: dict[str, Any],
        schema: type[ModelT],
    ) -> ModelT:
        """Call the agent and validate its reply against ``schema``."""
        request = render_agent_request(prompt, payload)
        schema_json = schema.model_json_schema()
        last_error: Exception | None = None
        for protocol_attempt in range(1, self.protocol_attempts + 1):
            try:
                response = self._invoke_with_transient_retry(request, schema_json)
            except subprocess.TimeoutExpired as error:
                raise StructuredCallError(
                    f"{self.provider} timed out after {self.timeout_s}s"
                ) from error
            except FileNotFoundError as error:
                raise StructuredCallError(
                    f"{self.provider} command {self.command!r} not found"
                ) from error

            try:
                return schema.model_validate(_decode_json(response))
            except (json.JSONDecodeError, ValidationError) as error:
                last_error = error
                if protocol_attempt < self.protocol_attempts:
                    request = _protocol_repair_request(
                        original_request=render_agent_request(prompt, payload),
                        previous_response=response,
                        error=error,
                        schema_name=schema.__name__,
                    )
                    continue
        if isinstance(last_error, ValidationError):
            raise StructuredCallError(
                f"{self.provider} reply violated {schema.__name__} after "
                f"{self.protocol_attempts} protocol attempts: "
                f"{last_error.error_count()} error(s); first: {last_error.errors()[0]}"
            ) from last_error
        raise StructuredCallError(
            f"{self.provider} returned non-JSON after {self.protocol_attempts} "
            f"protocol attempts: {last_error}"
        ) from last_error

    def _invoke_with_transient_retry(
        self, request: str, schema_json: dict[str, Any]
    ) -> str:
        """Serialize Codex calls across processes and retry service-side overloads.

        The Codex CLI retries a dropped stream internally, but a concurrency rejection
        still exits non-zero. Retrying that final result with jitter is useful only
        after releasing the shared slot, so another queued process can make progress.
        """
        attempts = max(1, self.transient_attempts)
        for attempt in range(1, attempts + 1):
            try:
                with self._concurrency_slot():
                    return self._invoke(request, schema_json)
            except StructuredCallError as error:
                if attempt == attempts or not _is_transient_provider_error(str(error)):
                    raise
                delay = self.transient_backoff_s * (2 ** (attempt - 1))
                time.sleep(delay + random.uniform(0, min(5.0, delay * 0.2)))
        raise AssertionError("unreachable")

    @contextmanager
    def _concurrency_slot(self):
        """Queue fairly across processes, then hold one external-request slot."""
        # Non-Codex providers do not share this account-level CLI capacity.
        max_concurrent_calls = int(
            os.environ.get("SLEEPGUIDE_CODEX_MAX_CONCURRENT_CALLS", self.max_concurrent_calls)
        )
        if self.protocol != "codex_exec" or max_concurrent_calls <= 0:
            yield
            return
        try:
            import fcntl
        except ImportError:  # pragma: no cover - all supported batch hosts are POSIX
            yield
            return

        lock_dir = self.concurrency_lock_dir or Path(
            os.environ.get("SLEEPGUIDE_CODEX_LOCK_DIR", "/tmp/sleepguide-codex-slots")
        )
        lock_dir.mkdir(parents=True, exist_ok=True)
        turnstile = (lock_dir / "queue.lock").open("a+", encoding="utf-8")
        handles = [
            (lock_dir / f"slot-{slot}.lock").open("a+", encoding="utf-8")
            for slot in range(max_concurrent_calls)
        ]
        acquired = None
        turnstile_held = False
        try:
            # Only the process at the front of this kernel-managed wait queue may
            # scan for a slot. Without the turnstile, a process that just released
            # a slot can repeatedly reacquire it before older pollers wake up.
            fcntl.flock(turnstile.fileno(), fcntl.LOCK_EX)
            turnstile_held = True
            while True:
                for handle in handles:
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        acquired = handle
                        break
                    except BlockingIOError:
                        continue
                if acquired is not None:
                    break
                time.sleep(0.25 + random.uniform(0, 0.15))
            fcntl.flock(turnstile.fileno(), fcntl.LOCK_UN)
            turnstile_held = False
            try:
                yield
            finally:
                fcntl.flock(acquired.fileno(), fcntl.LOCK_UN)
        finally:
            if turnstile_held:
                fcntl.flock(turnstile.fileno(), fcntl.LOCK_UN)
            turnstile.close()
            for handle in handles:
                handle.close()

    def _invoke(self, request: str, schema_json: dict[str, Any]) -> str:
        if self.protocol == "codex_exec":
            return self._invoke_codex(request, schema_json)
        if self.protocol == "claude_code":
            return self._invoke_claude(request, schema_json)
        return self._run([self.command, *self.args], request)

    def _invoke_codex(self, request: str, schema_json: dict[str, Any]) -> str:
        """Use Codex's native schema gate and read only its final message."""
        with tempfile.TemporaryDirectory(prefix="sleepguide-codex-") as directory:
            root = Path(directory)
            schema_path = root / "response_schema.json"
            output_path = root / "last_message.json"
            schema_path.write_text(
                json.dumps(_strict_response_schema(schema_json)), encoding="utf-8"
            )
            self._run(
                [
                    self.command,
                    *self.args,
                    "--output-schema",
                    str(schema_path),
                    "--output-last-message",
                    str(output_path),
                    "-",
                ],
                request,
            )
            if not output_path.exists():
                raise StructuredCallError(f"{self.provider} produced no final message")
            return output_path.read_text(encoding="utf-8")

    def _invoke_claude(self, request: str, schema_json: dict[str, Any]) -> str:
        """Use Claude Code's JSON-schema option; it returns a JSON result envelope."""
        return self._run(
            [self.command, *self.args, "--json-schema", json.dumps(schema_json)],
            request,
        )

    def _run(self, command: list[str], request: str) -> str:
        completed = subprocess.run(
            command,
            input=request,
            text=True,
            capture_output=True,
            check=False,
            timeout=self.timeout_s,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            if not stderr:
                # Some CLIs return structured API/authentication failures on stdout
                # while still using a non-zero exit code. Preserve that diagnostic.
                stderr = completed.stdout.strip()
            if len(stderr) > 4000:
                stderr = f"{stderr[:1000]}\n... [stderr truncated] ...\n{stderr[-3000:]}"
            raise StructuredCallError(f"{self.provider} exited {completed.returncode}: {stderr}")
        return completed.stdout


def render_agent_request(prompt: str, payload: dict[str, Any]) -> str:
    """Render the exact text sent to every structured runtime agent.

    Stage-2 export imports this function so SFT consumes the identical prompt framing,
    key order, indentation, and JSON encoding seen at inference time.  The outer JSONL
    record may be compact or sorted without changing this already-rendered string.
    """
    return "\n\n".join(
        [
            prompt.strip(),
            "Input payload (JSON):",
            json.dumps(payload, ensure_ascii=False, indent=2),
            "Return only the JSON object required by the supplied output schema.",
        ]
    )


def _protocol_repair_request(
    *,
    original_request: str,
    previous_response: str,
    error: Exception,
    schema_name: str,
) -> str:
    """Give a fresh agent one concrete malformed→correct protocol repair task."""

    return "\n\n".join(
        [
            original_request,
            "Protocol correction:",
            f"The previous {schema_name} reply failed validation: {error}",
            "Previous reply (content may be reused, field names and enums may not):",
            previous_response[:6000],
            (
                "Reason again from the same input, preserve valid substantive content, "
                "and return the exact canonical JSON shape shown in the role prompt. "
                "Do not invent semantically similar field names or enum values."
            ),
        ]
    )


def _strict_response_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Make a Pydantic schema satisfy Codex structured-output requirements.

    Codex requires every object property to be explicitly present in ``required``.
    Pydantic omits fields with defaults from that list even when an explicit empty
    value is preferable for an auditable artifact. Apply the rule recursively to
    root objects, definitions, array items, and nullable branches.
    """

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            # Codex rejects a JSON-schema property that combines a `$ref` with a
            # Pydantic-emitted `default`. Defaults are unnecessary here because this
            # function makes every response property required below; model defaults
            # still apply when artifacts are read back locally.
            node.pop("default", None)
            properties = node.get("properties")
            if isinstance(properties, dict):
                node["required"] = list(properties)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    rendered = json.loads(json.dumps(schema))
    visit(rendered)
    return rendered


def _decode_json(text: str) -> Any:
    """Decode a bare JSON reply or Claude Code's ``result`` envelope."""
    stripped = _strip_fences(text)
    decoder = json.JSONDecoder()
    parsed_values: list[Any] = []
    position = 0
    while position < len(stripped):
        while position < len(stripped) and stripped[position].isspace():
            position += 1
        if position >= len(stripped):
            break
        parsed, position = decoder.raw_decode(stripped, position)
        parsed_values.append(parsed)
    if not parsed_values:
        raise json.JSONDecodeError("empty JSON response", stripped, 0)
    # A resumed model occasionally emits an earlier object followed by its corrected
    # replacement. Accept this only when the entire response is a sequence of valid JSON
    # values, and use the last complete value as the intended final answer.
    parsed = parsed_values[-1]
    if isinstance(parsed, dict) and isinstance(parsed.get("result"), str):
        return _decode_json(parsed["result"])
    return parsed


def _strip_fences(text: str) -> str:
    """Tolerate ```json fences, which agents add regardless of instructions."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    body = stripped.split("\n", 1)[1] if "\n" in stripped else ""
    return body.rsplit("```", 1)[0].strip()


def _is_transient_provider_error(message: str) -> bool:
    """Recognize failures that can succeed later without changing the request."""
    normalized = message.lower()
    return any(
        marker in normalized
        for marker in (
            "concurrency limit exceeded",
            "rate limit",
            "too many requests",
            "stream disconnected",
            "connection reset",
            "connection aborted",
            "temporarily unavailable",
            "service unavailable",
            "gateway timeout",
        )
    )
