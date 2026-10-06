#!/usr/bin/env bash
# Fetch COCO 2017 instance annotations (for the clutter / object-count measure).
# Idempotent: skips if both instances files are already present.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ANN="$ROOT/data/coco_annotations"
mkdir -p "$ANN"

if [[ -f "$ANN/instances_train2017.json" && -f "$ANN/instances_val2017.json" ]]; then
  echo "COCO instance annotations already present in $ANN — skipping download."
  exit 0
fi

URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
TMP="$ANN/annotations_trainval2017.zip"
echo "Downloading $URL (~241 MB) ..."
curl -L --fail --retry 3 -o "$TMP" "$URL"
echo "Extracting instances files only ..."
unzip -o -j "$TMP" 'annotations/instances_train2017.json' 'annotations/instances_val2017.json' -d "$ANN"
rm -f "$TMP"
echo "Done:"
ls -1 "$ANN"
