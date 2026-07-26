from __future__ import annotations

import shlex
from typing import Any

SUPPORTED_PREFIXES = ("kubectl", "helm", "mongosh")

# Broad denylist for commands that are destructive, cluster-wide, interactive, or
# capable of arbitrary code/file execution. This checker is still conservative
# offline gating, not a substitute for a real sandbox/admission controller.
DANGEROUS_SUBSTRINGS = [
    "rm -rf",
    "curl ",
    "wget ",
    "| bash",
    "| sh",
    "bash -c",
    "sh -c",
    "python -c",
    "perl -e",
    "nc ",
    "netcat",
    "chmod 777",
    "> /etc/",
    "sudo ",
]

DENY_KUBECTL_VERBS = {
    "exec",
    "cp",
    "proxy",
    "port-forward",
    "attach",
    "debug",
    "auth",
    "create",
    "apply",
    "replace",
    "edit",
    "label",
    "annotate",
}

DENY_HELM_VERBS = {
    "uninstall",
    "delete",
    "install",
    "upgrade",
    "repo",
    "plugin",
}

DANGEROUS_DELETE_RESOURCES = {
    "namespace", "namespaces", "ns",
    "clusterrole", "clusterroles", "clusterrolebinding", "clusterrolebindings",
    "crd", "crds", "customresourcedefinition", "customresourcedefinitions",
    "pvc", "pv", "persistentvolumeclaim", "persistentvolumeclaims",
    "all",
    "secret", "secrets",
    "serviceaccount", "serviceaccounts",
}

ALLOWED_KUBECTL_VERBS = {"get", "describe", "logs", "rollout", "scale", "patch", "delete"}
ALLOWED_HELM_VERBS = {"rollback", "status", "history"}


def _split(cmd: str) -> list[str]:
    try:
        return shlex.split(str(cmd or ""))
    except Exception:
        return str(cmd or "").split()


def _has_shell_metacharacters(raw: str) -> bool:
    # Allow JSON patch brackets/quotes, but block shell control/operators that can
    # chain exfiltration or arbitrary execution.
    return any(x in raw for x in [";", "&&", "||", "`", "$(", "<(", ">("])


def _kubectl_safety(parts: list[str], raw: str) -> list[str]:
    reasons: list[str] = []
    if len(parts) < 2:
        return ["kubectl_missing_verb"]
    verb = parts[1]
    if verb in DENY_KUBECTL_VERBS:
        reasons.append(f"kubectl_denied_verb:{verb}")
    if verb not in ALLOWED_KUBECTL_VERBS:
        reasons.append(f"kubectl_unsupported_verb:{verb}")
    if verb == "delete":
        resource = parts[2].lower() if len(parts) > 2 else ""
        if resource in DANGEROUS_DELETE_RESOURCES:
            reasons.append(f"kubectl_dangerous_delete:{resource}")
        if "--all" in parts or "--all-namespaces" in parts or "-A" in parts:
            reasons.append("kubectl_delete_broad_scope")
    if verb == "patch":
        low = raw.lower()
        if "--type=json" not in low and "--type='json'" not in low and '--type="json"' not in low:
            reasons.append("kubectl_patch_requires_json_type")
        if "-p" not in parts and "--patch" not in parts and not any(p.startswith("-p=") or p.startswith("--patch=") for p in parts):
            reasons.append("kubectl_patch_missing_patch_payload")
    return reasons


def _helm_safety(parts: list[str]) -> list[str]:
    reasons: list[str] = []
    if len(parts) < 2:
        return ["helm_missing_verb"]
    verb = parts[1]
    if verb in DENY_HELM_VERBS:
        reasons.append(f"helm_denied_verb:{verb}")
    if verb not in ALLOWED_HELM_VERBS:
        reasons.append(f"helm_unsupported_verb:{verb}")
    return reasons


def check_command_safety(commands: list[str]) -> dict[str, Any]:
    unsafe = []
    for cmd in commands:
        raw = str(cmd or "").strip()
        low = raw.lower()
        parts = _split(raw)
        reasons: list[str] = []

        if not parts:
            reasons.append("empty_command")
        elif parts[0] not in SUPPORTED_PREFIXES:
            reasons.append("unsupported_prefix")

        if _has_shell_metacharacters(raw):
            reasons.append("shell_metacharacter")
        for pat in DANGEROUS_SUBSTRINGS:
            if pat in low:
                reasons.append(f"dangerous_substring:{pat.strip()}")

        if parts and parts[0] == "kubectl":
            reasons.extend(_kubectl_safety(parts, raw))
        elif parts and parts[0] == "helm":
            reasons.extend(_helm_safety(parts))
        elif parts and parts[0] == "mongosh":
            # mongosh is allowed only as a single-shot command, not an interactive shell.
            if "--eval" not in parts:
                reasons.append("mongosh_requires_eval")

        if reasons:
            unsafe.append({"command": cmd, "patterns": sorted(set(reasons))})

    return {"safe": not unsafe, "unsafe": unsafe, "num_commands": len(commands)}
