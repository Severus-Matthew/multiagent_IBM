from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable


@dataclass
class ApplicationTopology:
    edges: list[tuple[str, str]] = field(default_factory=list)
    entrypoints: list[str] = field(default_factory=list)
    source_files_scanned: int = 0
    cpp_services_discovered: int = 0
    lua_frontend_edges_discovered: int = 0
    evidence: dict[str, list[str]] = field(default_factory=dict)
    source_mode: str = "static_application_source_non_oracle"

    def to_dict(self) -> dict:
        out = asdict(self)
        out["edges"] = [{"src": s, "dst": d} for s, d in self.edges]
        return out


def _unique_edges(edges: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    return sorted({(str(s), str(d)) for s, d in edges if s and d and s != d})


def _infer_cpp_service_name(text: str, observable_services: set[str]) -> str | None:
    # DeathStarBench services normally register their tracer with the Kubernetes
    # service name.  This is preferable to guessing from directory names.
    patterns = [
        r'SetUpTracer\s*\([^,]+,\s*"([^"]+)"\s*\)',
        r'SetUpOpenTelemetryTracer\s*\(\s*"([^"]+)"\s*\)',
    ]
    for pattern in patterns:
        for match in re.findall(pattern, text):
            if match in observable_services:
                return match

    # Fallback: identify a service that reads its own listening port from the
    # shared service-config.  Require an exact observable service name.
    for match in re.findall(r'config_json\s*\[\s*"([^"]+)"\s*\]\s*\[\s*"port"\s*\]', text):
        if match in observable_services:
            return match
    return None


def _cpp_dependency_edges(app_root: Path, observable_services: set[str]) -> tuple[list[tuple[str, str]], dict[str, list[str]], int, int]:
    edges: list[tuple[str, str]] = []
    evidence: dict[str, list[str]] = {}
    files_scanned = 0
    callers = set()

    src_root = app_root / "src"
    if not src_root.exists():
        return edges, evidence, files_scanned, 0

    for path in sorted(src_root.rglob("*.cpp")):
        try:
            text = path.read_text(errors="ignore")
        except Exception:
            continue
        files_scanned += 1
        caller = _infer_cpp_service_name(text, observable_services)
        if not caller:
            continue
        callers.add(caller)

        refs = set(re.findall(r'config_json\s*\[\s*"([^"]+)"\s*\]', text))
        for dep in sorted(refs & observable_services):
            if dep == caller:
                continue
            edges.append((caller, dep))
            evidence.setdefault(f"{caller}->{dep}", []).append(str(path.relative_to(app_root)))

    return edges, evidence, files_scanned, len(callers)


def _frontend_service_name(observable_services: set[str]) -> str | None:
    # AIOpsLab's SocialNetwork Helm deployment exposes the nginx/OpenResty
    # frontend as nginx-thrift.  Prefer an exact observable frontend name and do
    # not fabricate a node when the deployment does not contain one.
    preferred = [
        "nginx-thrift",
        "nginx-web-server",
        "frontend",
    ]
    for name in preferred:
        if name in observable_services:
            return name
    nginx_like = sorted(s for s in observable_services if "nginx" in s.lower())
    return nginx_like[0] if len(nginx_like) == 1 else None


def _lua_frontend_edges(app_root: Path, observable_services: set[str]) -> tuple[list[tuple[str, str]], dict[str, list[str]], int, int, str | None]:
    edges: list[tuple[str, str]] = []
    evidence: dict[str, list[str]] = {}
    files_scanned = 0
    frontend = _frontend_service_name(observable_services)
    if not frontend:
        return edges, evidence, files_scanned, 0, None

    roots = [
        app_root / "nginx-web-server" / "lua-scripts",
        app_root / "openshift" / "nginx-thrift-config" / "lua-scripts",
    ]
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.lua")):
            try:
                text = path.read_text(errors="ignore")
            except Exception:
                continue
            files_scanned += 1

            # Handles patterns such as:
            #   connection(Client, "compose-post-service" .. k8s_suffix, 9090)
            # and other quoted service names used by the Lua clients.
            quoted = set(re.findall(r'"([a-z0-9][a-z0-9-]*service)"', text))
            for dep in sorted(quoted & observable_services):
                if dep == frontend:
                    continue
                edges.append((frontend, dep))
                evidence.setdefault(f"{frontend}->{dep}", []).append(str(path.relative_to(app_root)))

    return edges, evidence, files_scanned, len(edges), frontend


def discover_application_topology(
    application_source_root: str | Path,
    observable_services: Iterable[str],
) -> ApplicationTopology:
    """Derive non-oracle service topology from the application source tree.

    This catalog is independent of the injected fault, RCA labels, scenario name,
    and hidden fault context.  It is used only when a scenario's telemetry-derived
    graph is absent or incomplete.  For the AIOpsLab DeathStarBench application,
    C++ service-config references reveal downstream dependencies and the nginx Lua
    handlers reveal frontend-to-service request edges.
    """
    root = Path(application_source_root).expanduser().resolve()
    services = {str(s) for s in observable_services if s}
    if not root.exists():
        return ApplicationTopology(source_mode="static_application_source_missing")

    cpp_edges, cpp_evidence, cpp_files, cpp_callers = _cpp_dependency_edges(root, services)
    lua_edges, lua_evidence, lua_files, lua_count, frontend = _lua_frontend_edges(root, services)

    evidence = dict(cpp_evidence)
    for edge, files in lua_evidence.items():
        evidence.setdefault(edge, []).extend(files)

    edges = _unique_edges(cpp_edges + lua_edges)
    entrypoints = [frontend] if frontend and any(src == frontend for src, _ in edges) else []

    return ApplicationTopology(
        edges=edges,
        entrypoints=entrypoints,
        source_files_scanned=cpp_files + lua_files,
        cpp_services_discovered=cpp_callers,
        lua_frontend_edges_discovered=lua_count,
        evidence={k: sorted(set(v)) for k, v in sorted(evidence.items())},
        source_mode="static_application_source_non_oracle",
    )
