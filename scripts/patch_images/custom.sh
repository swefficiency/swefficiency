#!/usr/bin/env sh
set -eux

# git remote remove origin || true
# git tag -d $(git tag -l)
# git reflog expire --expire=now --all
# git gc --prune=now --aggressive

# Remove only tags pointing to commits after target timestamp
TARGET_TIMESTAMP=$(git show -s --format=%ci {base_commit})
git tag -l | while read tag; do TAG_COMMIT=$(git rev-list -n 1 "$tag"); TAG_TIME=$(git show -s --format=%ci "$TAG_COMMIT"); if [[ "$TAG_TIME" > "$TARGET_TIMESTAMP" ]]; then git tag -d "$tag"; fi; done
git reflog expire --expire=now --all
git gc --prune=now --aggressive