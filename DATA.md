# Data

No video frame, ground-truth mask or detection dump is redistributed here. The licences of
CDnet2014 (changedetection.net), LASIESTA (GTI, UPM) and BMC 2012 (backgroundmodelschallenge.eu)
are not recorded as permitting redistribution of derived data, so only the scripts that
regenerate the derived inputs and their SHA-256 are given.

The code reads the derived inputs below `$THS_DATASETS/_derived/` (default `./datasets`).

| File (under `_derived/`) | SHA-256 | Produced by |
|---|---|---|
| `06/inputs/gate_scores.parquet` | `0eb808e0a87972515eea82995f20549ffb8bc4516ace3e705b89287268992dcd` | logged gate score S_t of the corrected Deep3 re-run (public harness of Deep3, DOI 10.1109/JIOT.2026.3740931); provenance in `06/inputs/PROVENANCE.json` (`17b3f1e3…8a6b`) |
| `detections_cpu_yolo26s/{CDnet2014,LASIESTA,BMC}/*.json` | see `detections_cpu_yolo26s/MANIFEST.json` | always-on detector (yolo26s-seg, conf 0.25, imgsz 640) on the 121 videos; LASIESTA re-ordered by the numeric frame number |
| `06/traces_gt.parquet`, `06/traces_ge.parquet` | `b1d9db1b…d718`, `f2fbea8b…6eb6` | `code/policy_ref.py` (P0) from the two inputs above |
| `06/corpus_flat.npz` | `a8da51059d8ef253f6f2584a97c3c94a51159aa7ea19d351607542da111b1ec1` | `code/frontier_sweep.py::build_corpus()` from the two inputs above (121 videos, 166,589 frames, 3,170 events) |
| `06/shield_inputs.npz`, `06/shield_bench/*.u8` | regenerated on every run | `code/shield_bridge.py` |
