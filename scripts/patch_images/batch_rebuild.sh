#!/usr/bin/env bash
set -euo pipefail

# --- Config (env or flags) ----------------------------------------------------
: "${REPO:?Usage: export REPO=ghcr.io/swefficiency/swefficiency}"      # source repo
DEST_REPO="${DEST_REPO:-}"                                   # default = REPO
GITHUB_TOKEN="${GITHUB_TOKEN:-}"                             # PAT with read:packages (and write:packages for pushing)
WORKERS="${WORKERS:-16}"                                      # parallel workers
PREFIX="${PREFIX:-}"                                         # new tag prefix
SUFFIX="${SUFFIX:-}"                                # new tag suffix (default none)
FILTER_REGEX="${FILTER_REGEX:-}"                             # optional tag filter (eg. '^v1\.' )
DRY_RUN="${DRY_RUN:-0}"                                      # 1 = don't build/push, just print
SKIP_IF_EXISTS="${SKIP_IF_EXISTS:-0}"                        # 1 = skip if dest tag exists remotely
USE_PARALLEL="${USE_PARALLEL:-auto}"                         # auto|1|0
CUSTOM_SH="${CUSTOM_SH:-./scripts/patch_images/custom.sh}"                        # path to your commands file

# --- Basic checks -------------------------------------------------------------
if [[ ! -f "$CUSTOM_SH" ]]; then
  echo "custom.sh not found at $CUSTOM_SH" >&2; exit 1
fi

if [[ -z "$DEST_REPO" ]]; then
  DEST_REPO="$REPO"
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required." >&2; exit 1
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required." >&2; exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon not reachable." >&2; exit 1
fi

# --- Parse owner/image from REPO ----------------------------------------------
if [[ "$REPO" != ghcr.io/*/* ]]; then
  echo "REPO must look like ghcr.io/<owner>/<image>" >&2; exit 1
fi
R_NO_PREFIX="${REPO#ghcr.io/}"
OWNER="${R_NO_PREFIX%%/*}"
IMAGE="${R_NO_PREFIX#*/}"

D_NO_PREFIX="${DEST_REPO#ghcr.io/}"
DEST_OWNER="${D_NO_PREFIX%%/*}"
DEST_IMAGE="${D_NO_PREFIX#*/}"

# --- Get tags via GitHub Packages API -----------------------------------------
# Uses /orgs/:owner or /users/:owner (tries orgs first, then users)
API_BASE="https://api.github.com"
AUTH_HEADER=()
if [[ -n "${GITHUB_TOKEN}" ]]; then
  AUTH_HEADER=(-H "Authorization: Bearer ${GITHUB_TOKEN}")
fi

echo "Source: ghcr.io/${OWNER}/${IMAGE}"

get_tags_page() {
  local page="$1"
  # orgs endpoint
  curl -fsSL "${AUTH_HEADER[@]}" \
    "${API_BASE}/orgs/${OWNER}/packages/container/${IMAGE}/versions?per_page=100&page=${page}" \
    || return 1
}

get_tags_page_user() {
  local page="$1"
  # users endpoint
  curl -fsSL "${AUTH_HEADER[@]}" \
    "${API_BASE}/users/${OWNER}/packages/container/${IMAGE}/versions?per_page=100&page=${page}" \
    || return 1
}

echo "Fetching tags for ghcr.io/${OWNER}/${IMAGE} ..."
tmp_tags="$(mktemp)"
> "$tmp_tags"

page=1
mode=""
while : ; do
  if [[ -z "$mode" ]]; then
    # try org then user
    if json=$(get_tags_page "$page"); then
      mode="org"
    else
      if json=$(get_tags_page_user "$page"); then
        mode="user"
      else
        echo "Failed to list package versions. Check OWNER/IMAGE and permissions (read:packages)." >&2
        exit 1
      fi
    fi
  else
    if [[ "$mode" == "org" ]]; then
      json=$(get_tags_page "$page") || break
    else
      json=$(get_tags_page_user "$page") || break
    fi
  fi

  count=$(jq 'length' <<<"$json")
  if [[ "$count" -eq 0 ]]; then
    break
  fi

  # Each "version" can have multiple tags
  jq -r '.[]?.metadata?.container?.tags[]? // empty' <<<"$json" >> "$tmp_tags" || true
  ((page++))
done

if [[ ! -s "$tmp_tags" ]]; then
  echo "No tags found." >&2; exit 1
fi

# Dedup + filter
sort -u "$tmp_tags" | { if [[ -n "$FILTER_REGEX" ]]; then grep -E "$FILTER_REGEX"; else cat; fi; } > tags.txt
rm -f "$tmp_tags"

TAG_COUNT=$(wc -l < tags.txt | tr -d ' ')
echo "Found ${TAG_COUNT} tag(s). Saved to tags.txt."

# --- Build context (one-time) -------------------------------------------------
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

cat > "${BUILD_DIR}/Dockerfile" <<'EOF'
# build with BuildKit to allow ARG in FROM
# e.g. DOCKER_BUILDKIT=1 docker build --build-arg BASE_REF=... .
ARG BASE_REF
FROM ${BASE_REF}

# If your base needs bash, change the shebang in custom.sh and add it to the image there.
COPY custom.sh /opt/custom.sh
RUN chmod +x /opt/custom.sh && /opt/custom.sh && rm -f /opt/custom.sh

# Remove unnecessary files to reduce image size
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
EOF

cp "$CUSTOM_SH" "${BUILD_DIR}/custom.sh"

# --- Worker -------------------------------------------------------------------
build_one() {
  local tag="$1"
  local src_ref="ghcr.io/${OWNER}/${IMAGE}:${tag}"
  local new_tag="${PREFIX}${tag}${SUFFIX}"
  local dest_ref="ghcr.io/${DEST_OWNER}/${DEST_IMAGE}:${new_tag}"

  echo "[INFO] Processing ${src_ref} -> ${dest_ref}"

  if [[ "$SKIP_IF_EXISTS" == "1" ]]; then
    if docker manifest inspect "${dest_ref}" >/dev/null 2>&1; then
      echo "[SKIP] ${dest_ref} already exists."
      return 0
    fi
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY] Would build/push ${dest_ref} from ${src_ref}"
    return 0
  fi

  # Build new image by layering custom.sh
  DOCKER_BUILDKIT=1 docker build \
    --pull \
    --build-arg "BASE_REF=${src_ref}" \
    -t "${dest_ref}" \
    "${BUILD_DIR}" > /dev/null 2>&1

  # Push
  docker push "${dest_ref}" > /dev/null 2>&1

  # Remove image so we dont run out of space.
  docker image rm "${dest_ref}" || true > /dev/null 2>&1
  docker image rm "${src_ref}" || true > /dev/null 2>&1

  echo "[DONE] ${dest_ref}"
}

export -f build_one
export OWNER IMAGE DEST_OWNER DEST_IMAGE PREFIX SUFFIX SKIP_IF_EXISTS DRY_RUN BUILD_DIR

# --- Parallelize --------------------------------------------------------------
run_parallel() {
  if [[ "$USE_PARALLEL" == "1" || ( "$USE_PARALLEL" == "auto" && "$(command -v parallel >/dev/null 2>&1; echo $? )" -eq 0 ) ]]; then
    echo "Using GNU parallel with ${WORKERS} workers..."
    parallel --jobs "${WORKERS}" --joblog job.log --halt now,fail=1 build_one :::: tags.txt
  else
    echo "Using xargs -P ${WORKERS}..."
    # shellcheck disable=SC2016
    xargs -P "${WORKERS}" -n 1 -I{} bash -c 'build_one "$@"' _ {} < tags.txt
  fi
}

run_parallel

echo "All done."
