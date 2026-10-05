#!/usr/bin/env python3
# Merge a vision-only and an audio-only mmproj into one mixed-modality mmproj, so a single
# --mmproj serves both (e.g. jina-embeddings-v5-omni, which publishes them as two files).
# The per-modality projector types go to clip.vision.projector_type / clip.audio.projector_type.
#
#   python scripts/merge-mmproj.py vision-mmproj.gguf audio-mmproj.gguf out.gguf

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "gguf-py"))
import gguf  # noqa: E402

vision_path, audio_path, out_path = sys.argv[1:4]
readers = {"vision": gguf.GGUFReader(vision_path), "audio": gguf.GGUFReader(audio_path)}
writer = gguf.GGUFWriter(out_path, "clip")

seen = set()
for modality, r in readers.items():
    for key, field in r.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture" or key in seen:
            continue
        if key == "clip.projector_type":
            key = f"clip.{modality}.projector_type"
        seen.add(key)
        vtype = field.types[0]
        if vtype == gguf.GGUFValueType.ARRAY:
            writer.add_key_value(key, field.contents(), vtype, sub_type=field.types[-1])
        else:
            writer.add_key_value(key, field.contents(), vtype)

names = set()
for r in readers.values():
    for t in r.tensors:
        assert t.name not in names, f"tensor {t.name} is in both files"
        names.add(t.name)
        writer.add_tensor(t.name, t.data, raw_shape=list(reversed([int(d) for d in t.shape])), raw_dtype=t.tensor_type)

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
print(f"{out_path}: {len(names)} tensors")
