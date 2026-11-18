# Smoke check fixtures

Small output files for `test_smoke.py`, written once on 2026-11-18 by ComfyUI's
own save code at commit `05e878f3` with PyAV 17.0.1: the functions behind
`SaveAudioAdvanced`, `SaveVideo` and `SaveGLB`, fed with arrays made for the
purpose. No model ran. The files are as that code left them.

| File | What | Measured | Result |
|---|---|---|---|
| `audio_1s.flac` | 1.0 s of two tones and noise, 8,000 Hz, mono | 13,614 bytes a second | passes |
| `audio_quiet_1s.flac` | the same sound at 1 % volume | 7,108 bytes a second | passes |
| `audio_short.flac` | the first 0.3 s of the same sound | 0.3 s | fails: duration |
| `audio_silent_1s.flac` | silence, 1.0 s, 44,100 Hz, stereo | 142 bytes a second | fails: silent |
| `mesh_300.glb` | a sphere | 300 vertices, 560 faces | passes |
| `mesh_12.glb` | an icosahedron | 12 vertices, 20 faces | fails: size |
| `video_5f.mp4` | H.264, 640x352, 5 frames of a moving pattern | 5,034 | passes |
| `video_black_5f.mp4` | the same, all black | 352 | fails: blank |

The number of a video is its bytes a frame for each megapixel, over the frames
after the first.

Each FLAC file carries 8,192 bytes of padding that the writer adds. It is
metadata, so it is no part of "bytes a second".

PNG and text files are made inside the tests.

To write them again, call the same functions from a ComfyUI checkout:
`AudioSaveHelper.save_audio` (`comfy_api/latest/_ui.py`) with format `flac`,
`VideoFromComponents.save_to` (`comfy_api/latest/_input_impl/video_types.py`)
with container MP4 and codec H.264, and `save_glb`
(`comfy_extras/nodes_save_3d.py`). Another encoder version writes other
bytes: then correct the numbers that `test_smoke.py` pins.
