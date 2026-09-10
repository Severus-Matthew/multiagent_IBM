#!/usr/bin/env bash
# Supervised local port-forward for the generator's Prometheus endpoint
# (AIOpsLab's get_metrics action queries http://localhost:32000).
while true; do
  kubectl port-forward --address 127.0.0.1 svc/prometheus-server 32000:80 -n observe >/dev/null 2>&1
  sleep 2
done
