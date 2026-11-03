"""
Derive ``content_types_v1.json`` (beside this script) from the content-type
registry the Audace acquisition publishes (Audace-Lib
``tests/fixtures/content_types_v1.json``).

The copy is the producer's file with two changes, so that this public
package never spells a card model or vendor name:

* each former card variant tag (``cards[].legacy_tags``) is replaced by its
  64-bit FNV-1a digest, ``"fnv1a64:<16 hex digits>"`` — the digest the
  native core recognises those tags by (``src/content.rs``);
* the name of flag bit ``0x020`` is ``RESERVED`` (``FLAG_RESERVED``).

Nothing else changes; the result is written the way the producer writes its
file (``json.dumps(doc, indent=1, ensure_ascii=True) + "\\n"``), and
``tests/test_content_type.py`` checks that the copy is exactly this
derivation of the pinned upstream file. To follow a new upstream revision::

    python tests/assets/derive_content_types_v1.py ../Audace-Lib/tests/fixtures/content_types_v1.json

then put the two SHA-256 it prints in ``tests/test_content_type.py``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

#: Where the derived copy lives.
OUTPUT = Path(__file__).resolve().parent / "content_types_v1.json"

#: The neutral name of flag bit 0x020.
RESERVED_FLAG_NAME = "RESERVED"
RESERVED_FLAG_BIT = 0x020


def fnv1a64(data: bytes) -> int:
    """64-bit FNV-1a of ``data``."""
    h = 0xCBF2_9CE4_8422_2325
    for byte in data:
        h ^= byte
        h = (h * 0x0000_0100_0000_01B3) & 0xFFFF_FFFF_FFFF_FFFF
    return h


def tag_digest(tag: str) -> str:
    """How a former card variant tag is written in the derived copy."""
    return f"fnv1a64:{fnv1a64(tag.encode('ascii')):016x}"


def derive(upstream: bytes) -> bytes:
    """The derived copy of the producer's registry file ``upstream``."""
    doc = json.loads(upstream)
    for card in doc["cards"]:
        card["legacy_tags"] = [tag_digest(t) for t in card["legacy_tags"]]
    for flag in doc["flags"]:
        if flag["bit"] == RESERVED_FLAG_BIT:
            flag["name"] = RESERVED_FLAG_NAME
    return (json.dumps(doc, indent=1, ensure_ascii=True) + "\n").encode("ascii")


def main(argv: list) -> int:
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[-1].strip(), file=sys.stderr)
        return 2
    upstream = Path(argv[1]).read_bytes()
    derived = derive(upstream)
    OUTPUT.write_bytes(derived)
    print(f"upstream {hashlib.sha256(upstream).hexdigest()}")
    print(f"derived  {hashlib.sha256(derived).hexdigest()}  -> {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
