#!/usr/bin/env bash
# Build a corrected tokenizer directory for the azampatti INT4 checkpoint (upstream issue #2).
#
# The hub tokenizer.json carries a pre-tokenizer regex WITHOUT \p{M} (a transformers v5 bug, #49066, wrote it back
# when the checkpoint was saved), which over-tokenizes scripts with combining marks by 50-75%. The Qwen FP8 release
# ships the correct file with byte-identical vocab and merges, so copying it changes nothing else. The
# tokenizer_class override makes the engine's (still-buggy) Qwen2Tokenizer load the file instead of its own constant.
#
# usage: tools/apply_tokenizer_fix.sh [target-dir]    # default: $MODELS_DIR/tokenizer-fix
# then set TOKENIZER_DIR=<target-dir> in serve.sh settings (or the environment). Idempotent; needs network + python3.
set -eu
cd "$(dirname "$0")/.."; source ./config.env
TARGET="${1:-$MODELS_DIR/tokenizer-fix}"
SRC="https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/resolve/main/tokenizer.json"
[ -n "$MODEL_DIR" ] && [ -f "$MODEL_DIR/tokenizer_config.json" ] || { echo "model not found -- run ./setup.sh first" >&2; exit 1; }
mkdir -p "$TARGET"
python3 - "$TARGET" "$SRC" "$MODEL_DIR/tokenizer_config.json" <<'PY'
import json, os, sys, urllib.request
target, src, cfg = sys.argv[1:4]
tmp = os.path.join(target, "tokenizer.json.new")
with urllib.request.urlopen(src, timeout=180) as r, open(tmp, "wb") as f:
    while True:
        b = r.read(1 << 20)
        if not b:
            break
        f.write(b)
os.replace(tmp, os.path.join(target, "tokenizer.json"))
c = json.load(open(cfg))
c["tokenizer_class"] = "PreTrainedTokenizerFast"   # generic fast loader: honours the file, not a module constant
json.dump(c, open(os.path.join(target, "tokenizer_config.json"), "w"), ensure_ascii=False, indent=1)
print("wrote:", target)
PY
chmod 600 "$TARGET/tokenizer.json" "$TARGET/tokenizer_config.json"
# self-check inside the serving image when docker is up: the diacritized name must encode in 4 tokens, not 6
if docker info >/dev/null 2>&1 && [ -n "${IMAGE:-}" ] && docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker run --rm -v "$TARGET":"$TARGET":ro --entrypoint python3 "$IMAGE" -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('$TARGET')
n = len(t.encode('مُحَمَّد', add_special_tokens=False))
print('self-check مُحَمَّد:', n, 'tokens', 'OK' if n == 4 else 'UNEXPECTED -- check transformers version')
"
else
  echo "note: self-check skipped (docker or $IMAGE unavailable) -- verify via the API instead"
fi
