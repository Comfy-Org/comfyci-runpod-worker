# Importer fixtures

What `test_import_template.py` imports, and what must come out.

| File | What |
|---|---|
| `audio_ace_step_1_5_checkpoint.json`, `audio_minimax_music_3.json` | two templates, as `Comfy-Org/workflow_templates` has them at commit `85984c2b` |
| `object_info.json` | the definitions of the 16 node classes the two templates use, cut from a `/object_info` dump of ComfyUI `e9afbccc` |
| `ace_step_15_t2a_api.json`, `minimax_music3_t2a_api.json` | the workflow files the importer must write for them |
| `expected_entries.json` | their manifest entries |

The two workflow files and the entries were made by hand before the importer
could write an audio workflow. The importer has to give the same, with these
commands:

```
python scripts/import_template.py audio_ace_step_1_5_checkpoint --id ace_step_15_t2a \
    --cadence nightly --timeout-s 900 \
    --set 94.duration=30 --set 98.seconds=30 --expect seconds=30
python scripts/import_template.py audio_minimax_music_3 --id minimax_music3_t2a \
    --cadence weekly --timeout-s 1200 --set 37:13.max_duration=30
```

The sha256 and the size of each model file are what Hugging Face published on
2026-11-18. The test hands them to the importer: no test asks a host.
