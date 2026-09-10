from __future__ import annotations

"""Derive live fault mutations from the Twin's own objects, never from constants.

AIOpsLab's upstream injectors embed application-specific literals: the wrong-binary
injector only rewrites a container whose command already contains ``profile``, the
misconfig injector always installs one hard-coded image tag, and the MongoDB
adapters assume a ``mongodb-<name>`` naming convention with ``root``/``root``
credentials. Copying those literals into the Twin would inherit two defects at
once. It would reproduce their no-op behaviour whenever the requested target is
not the one literal they were written for, and it would bind the Twin to a single
application, registry and cluster.

Every mutation below is instead discovered from the rendered Twin bundle or from
live introspection of the Twin itself, so the same adapter works against a
different application, namespace or cluster. Discovery is deterministic: the same
bundle and target always yield the same mutation, which is required for a Twin
run to be reproducible. When the evidence needed for a mechanism is absent the
discovery fails closed with a machine-readable reason rather than silently
degrading to a weaker mutation.
"""

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

# RFC 2606 reserves `.invalid` and RFC 5735 reserves 240.0.0.0/4, so neither can
# resolve or route in any conforming environment. Using them keeps a corrupted
# endpoint unreachable without depending on local network topology.
UNROUTABLE_HOST = "aiops-twin-misconfigured.invalid"
UNROUTABLE_ADDRESS = "240.0.0.1"

CONFIG_KEY_MARKERS = (
    "addr", "host", "port", "url", "uri", "endpoint", "server", "conn", "dsn",
)

_MONGO_SHELLS = ("mongosh", "mongo")


class MutationDiscoveryError(RuntimeError):
    """Raised when the Twin does not contain the evidence a mechanism needs."""

    def __init__(self, reason: str, **details: Any) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details = details


CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _default_runner(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )


@dataclass
class LiveNamespaceBundle:
    """Bundle-shaped read-only view of a live namespace.

    The same discovery code then drives both the dataset generator (which mutates
    the real application namespace) and the Twin (which mutates rendered sparse
    manifests). Sharing one implementation is what makes a generated incident
    reproducible by the Twin: both sides compute the identical mutation from the
    identical evidence.
    """

    source_namespace: str
    target_namespace: str
    objects: list[dict[str, Any]] = field(default_factory=list)
    object_refs: list[dict[str, str]] = field(default_factory=list)
    read_only: bool = True


def live_namespace_bundle(
    namespace: str,
    *,
    kinds: Sequence[str] = ("deployments", "statefulsets", "configmaps", "services"),
    runner: CommandRunner | None = None,
) -> LiveNamespaceBundle:
    run = runner or _default_runner
    # kubectl omits per-item kind inside a List, so it is restored from the query.
    singular = {
        "deployments": "Deployment", "statefulsets": "StatefulSet",
        "configmaps": "ConfigMap", "services": "Service", "secrets": "Secret",
    }
    objects: list[dict[str, Any]] = []
    refs: list[dict[str, str]] = []
    for kind in kinds:
        proc = run(["kubectl", "get", kind, "-n", namespace, "-o", "json"])
        if proc.returncode != 0:
            continue
        try:
            payload = json.loads(proc.stdout or "{}")
        except json.JSONDecodeError:
            continue
        for item in payload.get("items", []) or []:
            if not isinstance(item, dict):
                continue
            resolved = dict(item)
            resolved.setdefault("kind", singular.get(kind, kind))
            if not resolved.get("kind"):
                resolved["kind"] = singular.get(kind, kind)
            objects.append(resolved)
            refs.append({
                "kind": str(resolved.get("kind") or ""),
                "name": str((resolved.get("metadata", {}) or {}).get("name") or ""),
            })
    return LiveNamespaceBundle(
        source_namespace=namespace, target_namespace=namespace,
        objects=objects, object_refs=refs,
    )


def _objects(bundle: Any, kind: str) -> list[dict[str, Any]]:
    return [
        obj for obj in getattr(bundle, "objects", []) or []
        if isinstance(obj, dict) and obj.get("kind") == kind
    ]


def _named(objects: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for obj in objects:
        if str((obj.get("metadata", {}) or {}).get("name") or "") == name:
            return obj
    return None


def workload_object(bundle: Any, service: str) -> dict[str, Any]:
    """Return the Deployment or StatefulSet that owns ``service`` in the Twin."""
    for kind in ("Deployment", "StatefulSet"):
        obj = _named(_objects(bundle, kind), service)
        if obj is not None:
            return obj
    raise MutationDiscoveryError(
        "target_workload_not_selected_in_twin", service=service
    )


def containers(workload: dict[str, Any]) -> list[dict[str, Any]]:
    spec = ((workload.get("spec", {}) or {}).get("template", {}) or {}).get("spec", {}) or {}
    return [c for c in (spec.get("containers", []) or []) if isinstance(c, dict)]


def primary_container(workload: dict[str, Any], service: str) -> dict[str, Any]:
    """Pick the container that represents the service.

    Prefer a name that matches the service (exactly, or as a suffix, which covers
    conventions such as ``hotel-reserv-<service>``); otherwise the first container.
    """
    rows = containers(workload)
    if not rows:
        raise MutationDiscoveryError(
            "target_workload_has_no_containers", service=service
        )
    low = service.strip().lower()
    for row in rows:
        if str(row.get("name") or "").strip().lower() == low:
            return row
    for row in rows:
        name = str(row.get("name") or "").strip().lower()
        if name.endswith(f"-{low}") or low.endswith(f"-{name}"):
            return row
    return rows[0]


# --------------------------------------------------------------------------
# wrong_binary
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BinarySwap:
    service: str
    container_name: str
    original_command: list[str] | None
    faulted_command: list[str]
    strategy: str
    provenance: dict[str, Any] = field(default_factory=dict)


def discover_binary_swap(bundle: Any, service: str) -> BinarySwap:
    """Choose a wrong—but real—entrypoint for ``service`` from its own siblings.

    Upstream AIOpsLab only rewrites containers whose command already contains
    ``profile``, so every other target is a no-op. Here the replacement command is
    drawn from another selected workload in the same Twin, which keeps the fault
    meaningful (a binary that exists in this application but is wrong for this
    service) without naming any application. When no sibling declares an explicit
    command the container's own image entrypoint is overridden with a deterministic
    name derived from the service, which still produces a genuine start failure.
    """
    workload = workload_object(bundle, service)
    container = primary_container(workload, service)
    original = list(container.get("command") or []) or None

    sibling_commands: dict[str, str] = {}
    for kind in ("Deployment", "StatefulSet"):
        for obj in _objects(bundle, kind):
            name = str((obj.get("metadata", {}) or {}).get("name") or "")
            if not name or name == service:
                continue
            for row in containers(obj):
                command = list(row.get("command") or [])
                if command and str(command[0]).strip():
                    sibling_commands.setdefault(str(command[0]).strip(), name)

    original_head = str(original[0]).strip() if original else ""
    candidates = sorted(k for k in sibling_commands if k != original_head)
    if candidates:
        chosen = candidates[0]
        return BinarySwap(
            service=service,
            container_name=str(container.get("name") or ""),
            original_command=original,
            faulted_command=[chosen],
            strategy="sibling_workload_entrypoint",
            provenance={
                "source_workload": sibling_commands[chosen],
                "candidate_commands": candidates,
                "selection_rule": "lexicographically_first_distinct_sibling_entrypoint",
            },
        )

    derived = f"{service.strip().lower()}-wrong-binary"
    return BinarySwap(
        service=service,
        container_name=str(container.get("name") or ""),
        original_command=original,
        faulted_command=[derived],
        strategy="derived_absent_entrypoint",
        provenance={
            "reason": "no_selected_sibling_declares_an_explicit_command",
            "selection_rule": "deterministic_name_derived_from_service",
        },
    )


# --------------------------------------------------------------------------
# application_config_misconfig
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigCorruption:
    service: str
    target_kind: str
    target_name: str
    key: str
    original_value: Any
    faulted_value: Any
    strategy: str
    container_name: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)


def _looks_like_endpoint(key: str, value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    low = key.strip().lower()
    return any(marker in low for marker in CONFIG_KEY_MARKERS)


def _corrupt_endpoint(value: str) -> str:
    """Redirect an endpoint at a reserved, unroutable destination.

    The shape of the original value is preserved so the application still parses
    its configuration and fails at connect time, which is the behaviour a real
    misconfiguration produces.
    """
    text = value.strip()
    if "://" in text:
        scheme, _, rest = text.partition("://")
        tail = rest.split("/", 1)
        port = ""
        hostpart = tail[0]
        if ":" in hostpart:
            port = ":" + hostpart.rsplit(":", 1)[1]
        path = "/" + tail[1] if len(tail) > 1 else ""
        return f"{scheme}://{UNROUTABLE_HOST}{port}{path}"
    if ":" in text and not text.startswith(":"):
        return f"{UNROUTABLE_HOST}:{text.rsplit(':', 1)[1]}"
    if text.replace(".", "").isdigit():
        return UNROUTABLE_ADDRESS
    return UNROUTABLE_HOST


_CONFIG_FILE_NAMES = ("config.json", "settings.json", "app.json")


def _service_tokens(service: str) -> set[str]:
    generic = {"service", "svc", "server", "api", "app", "mongodb", "mongo", "db", "redis", "memcached"}
    return {
        tok for tok in re.split(r"[-_./]+", service.strip().lower())
        if tok and tok not in generic
    }


def _endpoint_host(value: Any) -> str:
    """Host part of ``host:port`` or ``scheme://host:port/path`` values."""
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if "://" in text:
        text = text.partition("://")[2]
    text = text.split("/", 1)[0]
    if "@" in text:
        text = text.rsplit("@", 1)[1]
    return text.rsplit(":", 1)[0] if ":" in text and not text.startswith("[") else text


def _dockerfile_config_path(source_root: Path, relative: Path) -> tuple[str, dict[str, Any]]:
    """Resolve where the image places a source-tree file, from the Dockerfile.

    Only ``COPY <src> <dst>`` and ``WORKDIR`` instructions are consulted; the
    source tree that produced the image is the same one the static application
    topology is derived from, so this stays fault-independent.
    """
    dockerfile = source_root / "Dockerfile"
    if not dockerfile.is_file():
        raise MutationDiscoveryError("application_source_has_no_dockerfile", source_root=str(source_root))
    workdir = ""
    copies: list[tuple[str, str]] = []
    for raw in dockerfile.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if line.upper().startswith("WORKDIR "):
            workdir = line.split(None, 1)[1].strip()
        elif line.upper().startswith(("COPY ", "ADD ")):
            parts = [p for p in line.split()[1:] if not p.startswith("--")]
            if len(parts) >= 2:
                copies.append((parts[0], parts[-1]))
    rel = relative.as_posix()
    for src, dst in copies:
        src_norm = src.strip("./") or "."
        if src_norm in {".", ""}:
            base = dst if dst.endswith("/") or not workdir else dst
            path = f"{base.rstrip('/')}/{rel}"
            return path, {"dockerfile": str(dockerfile), "copy": [src, dst], "workdir": workdir}
        if src_norm == rel:
            path = dst if not dst.endswith("/") else f"{dst}{relative.name}"
            return path, {"dockerfile": str(dockerfile), "copy": [src, dst], "workdir": workdir}
    if workdir:
        return f"{workdir.rstrip('/')}/{rel}", {"dockerfile": str(dockerfile), "workdir": workdir, "copy": None}
    raise MutationDiscoveryError(
        "dockerfile_does_not_place_config_file", source_root=str(source_root), file=rel
    )


def discover_image_config_corruption(
    bundle: Any, service: str, application_source_root: str | Path
) -> ConfigCorruption:
    """Corrupt an endpoint in a configuration file baked into the service image.

    Some applications ship their configuration inside the image rather than via
    ConfigMap, environment or arguments (the DeathStarBench hotel services read a
    ``config.json`` copied in by their Dockerfile). The file is located in the
    application source tree, the key is chosen because its value names another
    workload present in this Twin and its key or value names the target service,
    and the in-container path comes from the Dockerfile. The mutation is applied
    by overlaying the corrupted file through a ConfigMap ``subPath`` mount, which
    is exactly the surface a real misconfiguration touches.
    """
    root = Path(application_source_root).expanduser().resolve()
    workload = workload_object(bundle, service)
    container = primary_container(workload, service)
    twin_services = {
        str((obj.get("metadata", {}) or {}).get("name") or "")
        for obj in _objects(bundle, "Service")
    }
    tokens = _service_tokens(service)
    # Keys the service's own entrypoint reads. A dependency endpoint that the
    # binary resolves at startup (registry, tracing backend) is a faithful
    # misconfiguration surface even when the key is not named after the service.
    main_text = ""
    for main_path in (root / "cmd" / service / "main.go", root / "cmd" / service.replace("-", "_") / "main.go"):
        if main_path.is_file():
            main_text = main_path.read_text(errors="ignore")
            break
    candidates: list[tuple[tuple[int, int, str, str], Path, dict[str, Any]]] = []
    for name in _CONFIG_FILE_NAMES:
        for path in sorted(root.glob(name)) + sorted(root.glob(f"*/{name}")):
            try:
                data = json.loads(path.read_text(errors="ignore"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            for key in sorted(data):
                value = data[key]
                if not _looks_like_endpoint(key, value):
                    continue
                host = _endpoint_host(value)
                if host not in twin_services:
                    continue
                key_tokens = set(re.findall(r"[a-z]+", key.lower()))
                host_tokens = _service_tokens(host)
                named = any(
                    kt.startswith(t) or t.startswith(kt)
                    for kt in key_tokens for t in tokens if len(t) >= 3 and len(kt) >= 3
                ) or bool(tokens & host_tokens)
                referenced_in_entrypoint = bool(main_text) and key in main_text
                if not named and not referenced_in_entrypoint:
                    continue
                # Tier 0: dependency named after the target service; tier 1: a
                # dependency the service's entrypoint resolves. Within a tier
                # prefer persistent datastores over caches (a cache miss is often
                # tolerated, a datastore failure is not), then key order.
                cache_like = any(marker in host.lower() for marker in ("memcache", "redis", "cache"))
                tier = 0 if named else 1
                rank = (tier, 0 if tokens & host_tokens else 1, 1 if cache_like else 0, key, str(path))
                candidates.append((rank, path, {"key": key, "value": value, "host": host,
                                                "selection_tier": "service_named" if named else "entrypoint_referenced"}))
    if not candidates:
        raise MutationDiscoveryError(
            "no_service_scoped_endpoint_in_image_config_file",
            service=service, source_root=str(root),
        )
    candidates.sort(key=lambda row: row[0])
    _, path, chosen = candidates[0]
    relative = path.relative_to(root)
    container_path, placement = _dockerfile_config_path(root, relative)
    return ConfigCorruption(
        service=service, target_kind="ConfigFile", target_name=container_path,
        key=str(chosen["key"]), original_value=chosen["value"],
        faulted_value=_corrupt_endpoint(str(chosen["value"])),
        strategy="image_embedded_config_file_endpoint",
        container_name=str(container.get("name") or ""),
        provenance={
            "source_file": str(path),
            "dependency_service": chosen["host"],
            "selection_tier": chosen["selection_tier"],
            "selection_rule": "service_named_endpoint_key_else_entrypoint_referenced_dependency_endpoint",
            "placement": placement,
            "candidate_keys": [row[2]["key"] for row in candidates],
        },
    )


def overlay_configmap_name(service: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", service.strip().lower()).strip("-")
    return f"twin-config-overlay-{slug}"[:63].rstrip("-")


def read_container_json_file(
    namespace: str, pod: str, container: str, path: str,
    runner: CommandRunner | None = None,
) -> dict[str, Any]:
    """Read and parse a JSON file from a running container; fails closed."""
    run = runner or _default_runner
    args = ["kubectl", "exec", "-n", namespace, pod]
    if container:
        args += ["-c", container]
    proc = run([*args, "--", "cat", path])
    if proc.returncode != 0:
        raise MutationDiscoveryError(
            "image_config_file_unreadable", pod=pod, path=path, stderr=proc.stderr.strip()[-500:]
        )
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise MutationDiscoveryError("image_config_file_not_json", pod=pod, path=path) from exc
    if not isinstance(data, dict):
        raise MutationDiscoveryError("image_config_file_not_an_object", pod=pod, path=path)
    return data


def config_file_overlay_objects(
    original_deployment: dict[str, Any],
    corruption: ConfigCorruption,
    namespace: str,
    live_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Render the ConfigMap and Deployment that apply an image-config corruption.

    The live file read from the running container must carry the discovered
    original value; otherwise the source tree and the image disagree and the
    mutation would not be the one that was discovered. The corrupted file is
    overlaid with a ``subPath`` mount so only that file changes.
    """
    if corruption.target_kind != "ConfigFile":
        raise ValueError("config_file_overlay_objects requires a ConfigFile corruption")
    observed = live_config.get(corruption.key)
    if observed != corruption.original_value:
        raise MutationDiscoveryError(
            "image_config_file_disagrees_with_source_tree",
            key=corruption.key, expected=corruption.original_value, observed=observed,
        )
    corrupted = dict(live_config)
    corrupted[corruption.key] = corruption.faulted_value
    file_name = corruption.target_name.rsplit("/", 1)[-1]
    cm_name = overlay_configmap_name(corruption.service)
    configmap = {
        "apiVersion": "v1", "kind": "ConfigMap",
        "metadata": {"name": cm_name, "namespace": namespace,
                     "labels": {"aiops.twin/fault-overlay": "application-config-misconfig"}},
        "data": {file_name: json.dumps(corrupted, indent=2, sort_keys=False) + "\n"},
    }
    faulted = json.loads(json.dumps(original_deployment))
    pod_spec = faulted["spec"]["template"]["spec"]
    volume_name = "twin-config-overlay"
    volumes = [v for v in (pod_spec.get("volumes") or []) if v.get("name") != volume_name]
    volumes.append({"name": volume_name, "configMap": {"name": cm_name}})
    pod_spec["volumes"] = volumes
    rows = pod_spec.get("containers") or []
    target = next(
        (row for row in rows if str(row.get("name") or "") == corruption.container_name),
        rows[0] if rows else None,
    )
    if target is None:
        raise MutationDiscoveryError("no_container_to_mutate", service=corruption.service)
    mounts = [m for m in (target.get("volumeMounts") or []) if m.get("name") != volume_name]
    mounts.append({
        "name": volume_name, "mountPath": corruption.target_name,
        "subPath": file_name, "readOnly": True,
    })
    target["volumeMounts"] = mounts
    return configmap, faulted


def resolve_application_source_root(bundle: Any, candidate_roots: Sequence[str | Path]) -> Path:
    """Pick the application source tree whose image config names this bundle's Services.

    Deterministic and fault-independent: the score is how many Service names in
    the bundle appear as endpoint hosts in a candidate's baked-in configuration.
    """
    twin_services = {
        str((obj.get("metadata", {}) or {}).get("name") or "")
        for obj in _objects(bundle, "Service")
    }
    scored: list[tuple[int, str, Path]] = []
    for candidate in candidate_roots:
        root = Path(candidate).expanduser().resolve()
        if not (root / "Dockerfile").is_file():
            continue
        hits = 0
        for name in _CONFIG_FILE_NAMES:
            for path in sorted(root.glob(name)) + sorted(root.glob(f"*/{name}")):
                try:
                    data = json.loads(path.read_text(errors="ignore"))
                except (OSError, json.JSONDecodeError):
                    continue
                if isinstance(data, dict):
                    hits += sum(
                        1 for key, value in data.items()
                        if _looks_like_endpoint(key, value) and _endpoint_host(value) in twin_services
                    )
        if hits:
            scored.append((hits, str(root), root))
    if not scored:
        raise MutationDiscoveryError(
            "no_application_source_root_matches_bundle_services",
            candidates=[str(c) for c in candidate_roots],
        )
    scored.sort(key=lambda row: (-row[0], row[1]))
    return scored[0][2]


def discover_config_corruption(
    bundle: Any, service: str, *, application_source_root: str | Path | None = None
) -> ConfigCorruption:
    """Find a real configuration value this service depends on and break it.

    Preference order is ConfigMap data, then container environment, then container
    arguments, because a mounted ConfigMap is the closest analogue to the
    configuration file a production misconfiguration would touch. When none of
    those carry an endpoint and an application source root is available, the
    configuration file baked into the image is used (see
    :func:`discover_image_config_corruption`). Nothing here references a specific
    image, registry or application.
    """
    workload = workload_object(bundle, service)
    container = primary_container(workload, service)

    mounted: list[str] = []
    pod_spec = ((workload.get("spec", {}) or {}).get("template", {}) or {}).get("spec", {}) or {}
    for volume in pod_spec.get("volumes", []) or []:
        name = ((volume or {}).get("configMap") or {}).get("name")
        if name:
            mounted.append(str(name))
    for row in containers(workload):
        for env_from in row.get("envFrom", []) or []:
            name = ((env_from or {}).get("configMapRef") or {}).get("name")
            if name:
                mounted.append(str(name))

    configmaps = _objects(bundle, "ConfigMap")
    for cm_name in sorted(set(mounted)):
        cm = _named(configmaps, cm_name)
        if cm is None:
            continue
        data = cm.get("data", {}) or {}
        for key in sorted(data):
            value = data[key]
            if _looks_like_endpoint(key, value):
                return ConfigCorruption(
                    service=service, target_kind="ConfigMap", target_name=cm_name,
                    key=key, original_value=value, faulted_value=_corrupt_endpoint(value),
                    strategy="mounted_configmap_endpoint",
                    provenance={"selection_rule": "first_endpoint_like_key_sorted"},
                )

    env_rows = [e for e in (container.get("env", []) or []) if isinstance(e, dict)]
    for entry in sorted(env_rows, key=lambda e: str(e.get("name") or "")):
        name = str(entry.get("name") or "")
        value = entry.get("value")
        if entry.get("valueFrom") is not None:
            continue
        if _looks_like_endpoint(name, value):
            return ConfigCorruption(
                service=service, target_kind="Env",
                target_name=str(workload.get("kind") or "Deployment"),
                key=name, original_value=value, faulted_value=_corrupt_endpoint(str(value)),
                strategy="container_environment_endpoint",
                container_name=str(container.get("name") or ""),
                provenance={"selection_rule": "first_endpoint_like_env_sorted"},
            )

    args = [str(a) for a in (container.get("args", []) or [])]
    for index, arg in enumerate(args):
        if "=" not in arg:
            continue
        flag, _, value = arg.partition("=")
        if _looks_like_endpoint(flag, value):
            return ConfigCorruption(
                service=service, target_kind="Args",
                target_name=str(workload.get("kind") or "Deployment"),
                key=str(index), original_value=arg,
                faulted_value=f"{flag}={_corrupt_endpoint(value)}",
                strategy="container_argument_endpoint",
                container_name=str(container.get("name") or ""),
                provenance={"selection_rule": "first_endpoint_like_argument"},
            )

    if application_source_root is not None:
        return discover_image_config_corruption(bundle, service, application_source_root)
    raise MutationDiscoveryError(
        "no_discoverable_configuration_endpoint_for_service", service=service
    )


# --------------------------------------------------------------------------
# MongoDB
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MongoAccess:
    service: str
    shell: str
    auth_enabled: bool
    admin_username: str
    admin_password: str
    auth_database: str
    application_databases: list[str]
    provenance: dict[str, Any] = field(default_factory=dict)

    def shell_command(self, script: str) -> list[str]:
        base = [self.shell, "--quiet"]
        if self.auth_enabled and self.admin_username:
            base += [
                "-u", self.admin_username, "-p", self.admin_password,
                "--authenticationDatabase", self.auth_database,
            ]
        return [*base, "--eval", script]


def _mongo_credentials(
    bundle: Any, workload: dict[str, Any]
) -> tuple[str, str, dict[str, Any]]:
    """Discover administrative credentials from declared workload resources.

    Mongo images commonly receive the root account through environment values,
    but the Hotel application creates it in a mounted init-script ConfigMap.
    Both are part of the workload's declared configuration, so inspecting them
    is portable and avoids application/service-name constants.
    """
    username = ""
    password = ""
    seen: dict[str, Any] = {}
    for row in containers(workload):
        for entry in row.get("env", []) or []:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "").upper()
            value = entry.get("value")
            if value is None:
                continue
            if "ROOT_USERNAME" in name or name.endswith("INITDB_ROOT_USERNAME"):
                username = str(value)
                seen["username_env"] = entry.get("name")
            elif "ROOT_PASSWORD" in name or name.endswith("INITDB_ROOT_PASSWORD"):
                password = str(value)
                seen["password_env"] = entry.get("name")
    if username and password:
        return username, password, seen

    pod_spec = (workload.get("spec", {}) or {}).get("template", {}).get("spec", {}) or {}
    mounted_configmaps: list[str] = []
    for volume in pod_spec.get("volumes", []) or []:
        name = ((volume or {}).get("configMap") or {}).get("name")
        if name:
            mounted_configmaps.append(str(name))

    # Accept only explicit shell-style variable assignments in resources mounted
    # by this workload. Values are used for authentication but never copied into
    # provenance or error details.
    assignments: dict[str, tuple[str, str]] = {}
    pattern = re.compile(
        r"(?m)^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
        r"(?:['\"]([^'\"]*)['\"]|([^\s#;]+))\s*(?:#.*)?$"
    )
    for cm_name in mounted_configmaps:
        cm = _named(_objects(bundle, "ConfigMap"), cm_name)
        if not cm:
            continue
        for data_key, text in ((cm.get("data", {}) or {}).items()):
            for match in pattern.finditer(str(text)):
                assignments[match.group(1).upper()] = (
                    match.group(2) if match.group(2) is not None else match.group(3),
                    f"{cm_name}/{data_key}",
                )

    user_keys = ("ROOT_USER", "MONGO_INITDB_ROOT_USERNAME", "ADMIN_USER")
    password_keys = ("ROOT_PWD", "ROOT_PASSWORD", "MONGO_INITDB_ROOT_PASSWORD", "ADMIN_PWD")
    chosen_user = next(((key, assignments[key]) for key in user_keys if key in assignments), None)
    chosen_password = next(
        ((key, assignments[key]) for key in password_keys if key in assignments), None
    )
    if chosen_user and chosen_password:
        seen.update({
            "credential_source": "mounted_configmap_init_script",
            "username_variable": chosen_user[0],
            "password_variable": chosen_password[0],
            "username_resource": chosen_user[1][1],
            "password_resource": chosen_password[1][1],
        })
        return chosen_user[1][0], chosen_password[1][0], seen
    return username, password, seen


def _auth_declared(workload: dict[str, Any]) -> bool:
    for row in containers(workload):
        tokens = [str(x) for x in (row.get("args") or [])] + [str(x) for x in (row.get("command") or [])]
        for token in tokens:
            if token.strip() in ("--auth",) or "authorization" in token.lower():
                return True
    return False


def discover_mongo_access(
    bundle: Any,
    service: str,
    namespace: str,
    pod: str,
    *,
    runner: CommandRunner | None = None,
) -> MongoAccess:
    """Introspect a live MongoDB in the Twin instead of assuming a convention.

    Upstream hardcodes the ``mongo`` shell, ``root``/``root`` credentials and a
    ``<service minus prefix>-db`` database name, which is why its adapters silently
    do nothing against any deployment that does not follow that convention. Here the
    shell, credentials, authorization state and database list all come from the
    running instance.
    """
    run = runner or _default_runner
    workload = workload_object(bundle, service)
    username, password, cred_provenance = _mongo_credentials(bundle, workload)
    declared_auth = _auth_declared(workload)

    shell = ""
    for candidate in _MONGO_SHELLS:
        probe = run(["kubectl", "exec", "-n", namespace, pod, "--",
                     "sh", "-c", f"command -v {candidate} >/dev/null 2>&1 && echo yes"])
        if probe.returncode == 0 and "yes" in (probe.stdout or ""):
            shell = candidate
            break
    if not shell:
        raise MutationDiscoveryError(
            "no_mongo_shell_available_in_target_container",
            service=service, probed=list(_MONGO_SHELLS),
        )

    def evaluate(script: str, *, authenticated: bool) -> subprocess.CompletedProcess[str]:
        base = [shell, "--quiet"]
        if authenticated and username:
            base += ["-u", username, "-p", password, "--authenticationDatabase", "admin"]
        return run(["kubectl", "exec", "-n", namespace, pod, "--", *base, "--eval", script])

    listing = "JSON.stringify(db.adminCommand({listDatabases:1}))"
    unauth = evaluate(listing, authenticated=False)
    auth_required = unauth.returncode != 0 or "requires authentication" in (
        (unauth.stdout or "") + (unauth.stderr or "")
    ).lower()

    result = evaluate(listing, authenticated=True) if auth_required else unauth
    databases: list[str] = []
    payload = (result.stdout or "").strip()
    start = payload.find("{")
    if start >= 0:
        try:
            parsed = json.loads(payload[start:])
            databases = sorted(
                str(row.get("name"))
                for row in (parsed.get("databases") or [])
                if row.get("name") and str(row.get("name")) not in
                ("admin", "config", "local")
            )
        except Exception:
            databases = []

    return MongoAccess(
        service=service,
        shell=shell,
        auth_enabled=bool(auth_required or declared_auth),
        admin_username=username,
        admin_password=password,
        auth_database="admin",
        application_databases=databases,
        provenance={
            "shell_probe_order": list(_MONGO_SHELLS),
            "authorization_declared_in_manifest": declared_auth,
            "authorization_enforced_by_server": bool(auth_required),
            **cred_provenance,
        },
    )


@dataclass(frozen=True)
class MongoUser:
    username: str
    database: str
    roles: list[dict[str, str]]

    def role_spec(self) -> str:
        return json.dumps(self.roles)


def mongo_eval(
    access: MongoAccess,
    namespace: str,
    pod: str,
    script: str,
    *,
    runner: CommandRunner | None = None,
) -> str:
    run = runner or _default_runner
    proc = run([
        "kubectl", "exec", "-n", namespace, pod, "--", *access.shell_command(script),
    ])
    if proc.returncode != 0:
        raise MutationDiscoveryError(
            "mongo_shell_command_failed",
            service=access.service, stderr=(proc.stderr or "")[-500:],
        )
    return proc.stdout or ""


def list_mongo_users(
    access: MongoAccess,
    namespace: str,
    pod: str,
    *,
    runner: CommandRunner | None = None,
) -> list[MongoUser]:
    """Read the real user catalog rather than assuming an ``admin`` account."""
    script = (
        "JSON.stringify(db.getSiblingDB('admin')"
        ".system.users.find({}, {user:1, db:1, roles:1}).toArray())"
    )
    payload = mongo_eval(access, namespace, pod, script, runner=runner).strip()
    start = payload.find("[")
    if start < 0:
        return []
    try:
        rows = json.loads(payload[start:])
    except Exception:
        return []
    users: list[MongoUser] = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("user"):
            continue
        users.append(MongoUser(
            username=str(row.get("user")),
            database=str(row.get("db") or "admin"),
            roles=[r for r in (row.get("roles") or []) if isinstance(r, dict)],
        ))
    return sorted(users, key=lambda u: (u.database, u.username))


def select_revocable_user(access: MongoAccess, users: list[MongoUser]) -> MongoUser:
    """Pick a user whose privileges can be withdrawn and later restored.

    The root credential the Twin authenticates with is excluded, otherwise the
    injection would lock the adapter out of its own restore path.
    """
    for user in users:
        if user.username == access.admin_username:
            continue
        if any(str(role.get("db")) in access.application_databases for role in user.roles):
            return user
    for user in users:
        if user.username != access.admin_username and user.roles:
            return user
    raise MutationDiscoveryError(
        "target_mongodb_has_no_revocable_application_user",
        service=access.service,
        known_users=[u.username for u in users],
    )


def require_auth_catalog(access: MongoAccess) -> None:
    """Fail closed when a mechanism needs a user catalog the target does not have.

    Several captured scenarios target MongoDB deployments that were never started
    with authorization enabled, so there is no user or role to revoke. Upstream
    silently skipped those, which is precisely why the recorded telemetry for them
    shows no fault.
    """
    if not access.auth_enabled:
        raise MutationDiscoveryError(
            "target_mongodb_has_no_authorization_enabled",
            service=access.service, provenance=access.provenance,
        )
    if not access.admin_username:
        raise MutationDiscoveryError(
            "target_mongodb_exposes_no_administrative_credential",
            service=access.service, provenance=access.provenance,
        )
