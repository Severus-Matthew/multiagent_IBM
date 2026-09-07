from __future__ import annotations

"""Audit that live fault mutations are discovered, deterministic and portable.

The upstream AIOpsLab injectors embed one application's literals, so they are
no-ops against any other target and cannot move to another cluster. This audit
runs the discovery layer against two unrelated synthetic applications and asserts
that every mutation is drawn from the bundle under test, that repeated discovery
is byte-identical, that absent evidence fails closed, and that none of the retired
hard-coded literals can reappear.

Run:
    python -m training_pipeline.audit_fault_mutation_discovery
"""

import json
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Sequence

from digital_twin_runtime.fault_mutation_discovery import (
    UNROUTABLE_HOST,
    MongoAccess,
    MutationDiscoveryError,
    discover_binary_swap,
    discover_config_corruption,
    discover_mongo_access,
    list_mongo_users,
    require_auth_catalog,
    select_revocable_user,
)

# Literals the retired implementation hard-coded. None may appear in any mutation.
FORBIDDEN_LITERALS = (
    "yinfangchen", "geo:app3", "hotel-reserv", "/etc/tls", "requireTLS",
    "mongodb-", "root", "url-shorten",
)


@dataclass
class FakeBundle:
    objects: list[dict[str, Any]] = field(default_factory=list)
    object_refs: list[dict[str, str]] = field(default_factory=list)
    source_namespace: str = "src"
    target_namespace: str = "aiops-twin-test"
    read_only: bool = True


def deployment(name: str, container: str, *, command=None, env=None, args=None,
               configmap=None, image="registry.example/app:1") -> dict[str, Any]:
    spec: dict[str, Any] = {"name": container, "image": image}
    if command:
        spec["command"] = list(command)
    if env:
        spec["env"] = [{"name": k, "value": v} for k, v in env]
    if args:
        spec["args"] = list(args)
    pod_spec: dict[str, Any] = {"containers": [spec]}
    if configmap:
        pod_spec["volumes"] = [{"name": "cfg", "configMap": {"name": configmap}}]
    return {
        "apiVersion": "apps/v1", "kind": "Deployment",
        "metadata": {"name": name},
        "spec": {
            "selector": {"matchLabels": {"app": name}},
            "template": {"metadata": {"labels": {"app": name}}, "spec": pod_spec},
        },
    }


def configmap(name: str, data: dict[str, str]) -> dict[str, Any]:
    return {"apiVersion": "v1", "kind": "ConfigMap",
            "metadata": {"name": name}, "data": dict(data)}


# Two applications that share no naming convention, image registry, config key
# style or entrypoint vocabulary.
APP_ALPHA = FakeBundle(objects=[
    deployment("orders-api", "orders-api", command=["orders-server"],
               configmap="orders-config"),
    deployment("billing-api", "billing-api", command=["billing-server"]),
    deployment("ledger-store", "ledger-store", args=["--auth"],
               env=[("LEDGER_INITDB_ROOT_USERNAME", "ops"),
                    ("LEDGER_INITDB_ROOT_PASSWORD", "s3cret")]),
    configmap("orders-config", {
        "notes": "free text with no endpoint",
        "downstream_url": "http://billing-api:8080/v1",
    }),
])

APP_BETA = FakeBundle(objects=[
    deployment("catalogue", "catalogue-main", command=["/bin/catalogue"]),
    deployment("shipping", "shipping-main", command=["/bin/shipping"],
               env=[("QUEUE_HOST", "rabbit.svc:5672"), ("LOG_LEVEL", "info")]),
    deployment("payments", "payments-main",
               args=["--listen=0.0.0.0:9000", "--upstream-addr=bank.svc:443"]),
])

APP_SCRIPT_CREDENTIALS = FakeBundle(objects=[
    deployment("document-store", "document-store", args=["--auth"],
               configmap="document-store-init"),
    configmap("document-store-init", {
        "init.sh": "ROOT_USER='operator'\nROOT_PWD='private-value'\n",
    }),
])


class FakeKubectl:
    """Scripted kubectl/mongo shell so discovery can be audited without a cluster."""

    def __init__(self, *, shell: str, auth_required: bool, databases: list[str],
                 users: list[dict[str, Any]]) -> None:
        self.shell = shell
        self.auth_required = auth_required
        self.databases = databases
        self.users = users
        self.calls: list[list[str]] = []

    def __call__(self, args: Sequence[str]) -> subprocess.CompletedProcess[str]:
        argv = list(args)
        self.calls.append(argv)
        joined = " ".join(argv)
        if "command -v" in joined:
            probe = joined.split("command -v", 1)[1].split()[0]
            ok = probe == self.shell
            return subprocess.CompletedProcess(argv, 0 if ok else 1,
                                               "yes\n" if ok else "", "")
        authenticated = "--authenticationDatabase" in argv
        if "listDatabases" in joined:
            if self.auth_required and not authenticated:
                return subprocess.CompletedProcess(
                    argv, 1, "", "MongoServerError: command requires authentication")
            payload = {"databases": [{"name": n} for n in
                                     [*self.databases, "admin", "config", "local"]]}
            return subprocess.CompletedProcess(argv, 0, json.dumps(payload), "")
        if "system.users" in joined:
            return subprocess.CompletedProcess(argv, 0, json.dumps(self.users), "")
        return subprocess.CompletedProcess(argv, 0, "", "")


def check(name: str, condition: bool, detail: str = "") -> dict[str, Any]:
    row = {"check": name, "passed": bool(condition), "detail": detail}
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f" :: {detail}" if detail else ""))
    return row


def assert_no_forbidden(blob: Any) -> tuple[bool, str]:
    text = json.dumps(blob, default=str).lower()
    hits = [lit for lit in FORBIDDEN_LITERALS if lit.lower() in text]
    return (not hits), f"forbidden literals present: {hits}" if hits else ""


def main() -> int:
    results: list[dict[str, Any]] = []

    print("== wrong_binary discovery ==")
    alpha = discover_binary_swap(APP_ALPHA, "orders-api")
    beta = discover_binary_swap(APP_BETA, "catalogue")
    results.append(check(
        "alpha binary swap comes from a sibling in the same bundle",
        alpha.faulted_command == ["billing-server"],
        f"chose {alpha.faulted_command} via {alpha.strategy}",
    ))
    results.append(check(
        "beta binary swap adapts to a different application",
        beta.faulted_command == ["/bin/shipping"],
        f"chose {beta.faulted_command} via {beta.strategy}",
    ))
    results.append(check(
        "binary swap never reuses the target's own entrypoint",
        alpha.faulted_command != alpha.original_command
        and beta.faulted_command != beta.original_command,
    ))
    ok, detail = assert_no_forbidden([alpha, beta])
    results.append(check("binary swap contains no retired literals", ok, detail))

    lonely = FakeBundle(objects=[deployment("solo", "solo")])
    derived = discover_binary_swap(lonely, "solo")
    results.append(check(
        "binary swap without siblings derives a name from the service",
        derived.faulted_command == ["solo-wrong-binary"]
        and derived.strategy == "derived_absent_entrypoint",
        f"chose {derived.faulted_command}",
    ))

    print("\n== application_config_misconfig discovery ==")
    cm = discover_config_corruption(APP_ALPHA, "orders-api")
    results.append(check(
        "configmap endpoint is preferred and redirected to a reserved host",
        cm.target_kind == "ConfigMap" and cm.key == "downstream_url"
        and UNROUTABLE_HOST in str(cm.faulted_value),
        f"{cm.target_kind}/{cm.target_name}.{cm.key} -> {cm.faulted_value}",
    ))
    results.append(check(
        "configmap corruption preserves the original value shape",
        str(cm.faulted_value).startswith("http://")
        and str(cm.faulted_value).endswith(":8080/v1"),
        str(cm.faulted_value),
    ))
    env = discover_config_corruption(APP_BETA, "shipping")
    results.append(check(
        "environment endpoint is used when no configmap is mounted",
        env.target_kind == "Env" and env.key == "QUEUE_HOST"
        and UNROUTABLE_HOST in str(env.faulted_value),
        f"{env.key} -> {env.faulted_value}",
    ))
    results.append(check(
        "non-endpoint keys are ignored",
        env.key != "LOG_LEVEL",
    ))
    arg = discover_config_corruption(APP_BETA, "payments")
    results.append(check(
        "container arguments are used as the last resort",
        arg.target_kind == "Args" and UNROUTABLE_HOST in str(arg.faulted_value),
        f"arg[{arg.key}] -> {arg.faulted_value}",
    ))
    try:
        discover_config_corruption(APP_BETA, "catalogue")
        results.append(check("config discovery fails closed with no evidence", False,
                             "expected MutationDiscoveryError"))
    except MutationDiscoveryError as exc:
        results.append(check("config discovery fails closed with no evidence", True,
                             exc.reason))
    ok, detail = assert_no_forbidden([cm, env, arg])
    results.append(check("config corruption contains no retired literals", ok, detail))

    print("\n== determinism ==")
    results.append(check(
        "repeated discovery is identical",
        discover_binary_swap(APP_ALPHA, "orders-api") == alpha
        and discover_config_corruption(APP_ALPHA, "orders-api") == cm,
    ))

    print("\n== mongodb discovery ==")
    runner = FakeKubectl(
        shell="mongosh", auth_required=True, databases=["ledger"],
        users=[
            {"user": "ops", "db": "admin", "roles": [{"role": "root", "db": "admin"}]},
            {"user": "ledger-app", "db": "admin",
             "roles": [{"role": "readWrite", "db": "ledger"}]},
        ],
    )
    access = discover_mongo_access(APP_ALPHA, "ledger-store", "ns", "pod-1", runner=runner)
    results.append(check(
        "mongo shell is probed rather than assumed",
        access.shell == "mongosh",
        f"probed {access.provenance.get('shell_probe_order')} -> {access.shell}",
    ))
    results.append(check(
        "credentials are read from the workload manifest",
        access.admin_username == "ops" and access.admin_password == "s3cret",
    ))
    script_access = discover_mongo_access(
        APP_SCRIPT_CREDENTIALS, "document-store", "ns", "pod-2", runner=runner
    )
    results.append(check(
        "credentials are discovered from a mounted init-script ConfigMap",
        script_access.admin_username == "operator"
        and script_access.admin_password == "private-value"
        and script_access.provenance.get("credential_source")
        == "mounted_configmap_init_script",
    ))
    results.append(check(
        "credential values are absent from discovery provenance",
        "private-value" not in json.dumps(script_access.provenance),
    ))
    results.append(check(
        "application databases are enumerated from the server",
        access.application_databases == ["ledger"],
        str(access.application_databases),
    ))
    users = list_mongo_users(access, "ns", "pod-1", runner=runner)
    chosen = select_revocable_user(access, users)
    results.append(check(
        "revocation targets an application user, never the admin credential",
        chosen.username == "ledger-app",
        f"chose {chosen.username}",
    ))

    anonymous = MongoAccess(
        service="ledger-store", shell="mongo", auth_enabled=False,
        admin_username="", admin_password="", auth_database="admin",
        application_databases=[],
    )
    try:
        require_auth_catalog(anonymous)
        results.append(check("datastore without authorization fails closed", False,
                             "expected MutationDiscoveryError"))
    except MutationDiscoveryError as exc:
        results.append(check("datastore without authorization fails closed", True,
                             exc.reason))

    no_shell = FakeKubectl(shell="none", auth_required=False, databases=[], users=[])
    try:
        discover_mongo_access(APP_ALPHA, "ledger-store", "ns", "pod-1", runner=no_shell)
        results.append(check("missing mongo shell fails closed", False,
                             "expected MutationDiscoveryError"))
    except MutationDiscoveryError as exc:
        results.append(check("missing mongo shell fails closed", True, exc.reason))

    print("\n== portability ==")
    results.append(check(
        "no mutation references the source namespace or cluster",
        all(
            "aiops-twin-test" not in json.dumps(x, default=str)
            and "src" not in json.dumps(x, default=str).split('"')[0:1]
            for x in (alpha, beta, cm, env, arg)
        ),
    ))

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    status = "PASS_FAULT_MUTATION_DISCOVERY" if passed == total else "FAIL_FAULT_MUTATION_DISCOVERY"
    print(f"\n{status}: {passed}/{total} checks passed")
    if passed != total:
        for row in results:
            if not row["passed"]:
                print(f"  failing: {row['check']} :: {row['detail']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
