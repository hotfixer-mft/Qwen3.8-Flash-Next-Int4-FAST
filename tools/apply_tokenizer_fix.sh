#!/usr/bin/env bash
# Build a corrected tokenizer directory for the azampatti INT4 checkpoint (upstream issue #2).
#
# The hub tokenizer.json carries a pre-tokenizer regex WITHOUT \p{M} (a transformers v5 bug, #49066, wrote it back
# when the checkpoint was saved), which over-tokenizes scripts with combining marks by 50-75%. The Qwen FP8 release
# ships the correct file with byte-identical vocab and merges, so copying it changes nothing else. The
# tokenizer_class override makes the engine's (still-buggy) Qwen2Tokenizer load the file instead of its own constant.
#
# The fix dir also carries chat_template.jinja, copied from the model dir: --tokenizer replaces ALL tokenizer
# files the model would otherwise load, and the hub INT4 snapshot keeps its template nowhere else -- a fix dir
# without it serves /v1/completions fine and answers 400 "no chat template" on every /v1/chat/completions call
# (measured 2026-10-19, 12 min of chat outage after a hand-rebuilt fix dir lost the template).
#
# usage: tools/apply_tokenizer_fix.sh [target-dir]           # build (default: $MODELS_DIR/tokenizer-fix)
#        tools/apply_tokenizer_fix.sh --check [target-dir]   # verify an existing fix dir, change nothing
# then set TOKENIZER_DIR=<target-dir> in serve.sh settings (or the environment). Building is idempotent and needs
# network + python3; --check is read-only and needs only python3. NOTE (v16b layout): the serving container only
# mounts the model dir -- place the target INSIDE $MODEL_DIR (and point --tokenizer at it as a /model/... path),
# or the tokenizer dir is invisible in the container and the engine dies at boot with "Repo id must be in the form...".
set -eu
cd "$(dirname "$0")/.."; source ./config.env

CHECK=0
[ "${1:-}" = "--check" ] && { CHECK=1; shift; }
TARGET="${1:-$MODELS_DIR/tokenizer-fix}"

validate() {  # one line per file, rc!=0 if the dir is not a usable fix dir
  python3 - "$1" <<'PY'
import json, os, sys
d, bad = sys.argv[1], 0
p = os.path.join(d, "tokenizer.json")
try:
    t = json.load(open(p))
    m = t["model"]
    assert m["type"] == "BPE" and len(m["vocab"]) > 1000, "unexpected model"
    assert "\\p{M}" in json.dumps(t["pre_tokenizer"]), "issue-#2 regression: pre-tokenizer regex lost \\p{M}"
    print("ok   tokenizer.json (%d B, %d vocab, \\p{M} present)" % (os.path.getsize(p), len(m["vocab"])))
except Exception as e:
    print("FAIL tokenizer.json:", e); bad = 1
p = os.path.join(d, "tokenizer_config.json")
try:
    k = json.load(open(p)).get("tokenizer_class")
    if k != "PreTrainedTokenizerFast":
        print("FAIL tokenizer_config.json: tokenizer_class is %r, want 'PreTrainedTokenizerFast'" % k); bad = 1
    else:
        print("ok   tokenizer_config.json (tokenizer_class=%s)" % k)
except Exception as e:
    print("FAIL tokenizer_config.json:", e); bad = 1
p = os.path.join(d, "chat_template.jinja")
try:
    n = os.path.getsize(p); assert n > 100, "empty"
    print("ok   chat_template.jinja (%d B)" % n)
except Exception as e:
    print("FAIL chat_template.jinja:", e); bad = 1
sys.exit(bad)
PY
}

if [ "$CHECK" = 1 ]; then
  if validate "$TARGET"; then echo "check: PASS $TARGET"
  else echo "check: FAIL $TARGET -- run tools/apply_tokenizer_fix.sh to rebuild it" >&2; exit 1; fi
  exit 0
fi

SRC="https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/resolve/main/tokenizer.json"
[ -n "$MODEL_DIR" ] && [ -f "$MODEL_DIR/tokenizer_config.json" ] || { echo "model not found -- run ./setup.sh first" >&2; exit 1; }
[ -f "$MODEL_DIR/chat_template.jinja" ] || { echo "$MODEL_DIR has no chat_template.jinja -- point MODEL_DIR at the served model dir" >&2; exit 1; }
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
PY
cp "$MODEL_DIR/chat_template.jinja" "$TARGET/chat_template.jinja"
chmod 600 "$TARGET/tokenizer.json" "$TARGET/tokenizer_config.json" "$TARGET/chat_template.jinja"
validate "$TARGET" || { echo "apply: wrote an INVALID fix dir -- refusing to call it done" >&2; exit 1; }

# self-check inside the serving image when docker is up: the diacritized name must encode in 4 tokens, not 6,
# and the loaded tokenizer must expose the chat template
if docker info >/dev/null 2>&1 && [ -n "${IMAGE:-}" ] && docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker run --rm -v "$TARGET":"$TARGET":ro --entrypoint python3 "$IMAGE" -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('$TARGET')
n = len(t.encode('مُحَمَّد', add_special_tokens=False))
print('self-check مُحَمَّد:', n, 'tokens', 'OK' if n == 4 else 'UNEXPECTED -- check transformers version')
ct = getattr(t, 'chat_template', None) or ''
print('self-check chat_template:', 'OK (%d B)' % len(ct) if ct else 'MISSING -- chat requests will 400')
"
else
  echo "note: in-image self-check skipped (docker or ${IMAGE:-\$IMAGE} unavailable) -- verify via the API instead"
fi
echo "wrote: $TARGET -- now set TOKENIZER_DIR=$TARGET"
