#!/bin/bash

ROOT_DIR=data

export extract_archive
extract_archive() {
    archive="$1"
    dir="$(dirname "$archive")"
    basename=$(basename "$archive" .tar)
    if [ -d "$dir/$basename" ]; then
        echo "Directory $dir/$basename already exists, skipping."
        return
    fi
    echo "Extracting $archive in $dir"
    case "$archive" in
        *.tar)    tar -xf "$archive" -C "$dir" ;;
    esac
}

export -f extract_archive

find "$ROOT_DIR" -type f \( -name "*.tar" -o -name "*.tar.gz" -o -name "*.tgz" \) \
    | grep "pandas" | parallel --jobs 8 extract_archive {}