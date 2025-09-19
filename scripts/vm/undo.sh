sudo bash -c 'set -euo pipefail;
TOTAL=$(nproc --all || echo 0); RANGE=$([ "$TOTAL" -gt 0 ] && echo "0-$((TOTAL-1))" || echo "");
# 1) Reset live CPU affinity for dockerd to all CPUs (if running)
if [[ -n "$RANGE" ]]; then
  for p in $(pgrep -x dockerd || true); do taskset -pc "$RANGE" "$p" || true; done
fi
# 2) Remove dockerd/containerd drop-ins
for u in docker.service docker.io.service moby-engine.service snap.docker.dockerd.service dockerd.service; do
  d="/etc/systemd/system/$u.d"; f="$d/cpu-mem-limits.conf";
  if [[ -f "$f" ]]; then rm -f "$f"; rmdir --ignore-fail-on-non-empty "$d" || true; fi
done
d="/etc/systemd/system/containerd.service.d"; f="$d/cpu-mem-limits.conf";
if [[ -f "$f" ]]; then rm -f "$f"; rmdir --ignore-fail-on-non-empty "$d" || true; fi
# 3) Disable & remove the boot-time affinity oneshot (if it was created)
if systemctl list-unit-files | awk '"'"'{print $1}'"'"' | grep -qx pin-dockerd-affinity.service; then
  systemctl disable --now pin-dockerd-affinity.service || true
fi
rm -f /etc/systemd/system/pin-dockerd-affinity.service || true
# 4) Reload systemd & restart daemons (whichever exist)
systemctl daemon-reload
systemctl try-restart docker.service docker.io.service moby-engine.service snap.docker.dockerd.service dockerd.service 2>/dev/null || true
systemctl try-restart containerd.service 2>/dev/null || true
echo "Reverted dockerd/containerd CPU affinity & memory limits (drop-ins removed, oneshot disabled, services reloaded)."
'
#!/bin/bash