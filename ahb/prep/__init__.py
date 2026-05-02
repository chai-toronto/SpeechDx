"""Manifest preparation modules — one per dataset and per cross-pair.

Each module exposes one or more ``prepare_*`` functions invoked by
``ahb.prep.dispatch.ensure_manifest``. Old task YAMLs reference the legacy
``training.dataio.prep_*`` paths; ``dispatch`` maps those to the modules
in this package via :data:`OLD_TO_NEW_MODULE`.
"""

from __future__ import annotations

OLD_TO_NEW_MODULE = {
    # Single-dataset
    "training.dataio.prep_default":         "ahb.prep.standard",
    "training.dataio.prep_aphasia":         "ahb.prep.aphasia",
    "training.dataio.prep_avfad":           "ahb.prep.avfad",
    "training.dataio.prep_c9s":             "ahb.prep.c9s",
    "training.dataio.prep_coswara":         "ahb.prep.coswara",
    "training.dataio.prep_dbank":           "ahb.prep.dbank",
    "training.dataio.prep_edaic":           "ahb.prep.edaic",
    "training.dataio.prep_iemocap":         "ahb.prep.iemocap",
    "training.dataio.prep_ksof":            "ahb.prep.ksof",
    "training.dataio.prep_mvdr":            "ahb.prep.mvdr",
    "training.dataio.prep_nemours":         "ahb.prep.nemours",
    "training.dataio.prep_ravdess":         "ahb.prep.ravdess",
    "training.dataio.prep_torgo":           "ahb.prep.torgo",
    "training.dataio.prep_uaspeech":        "ahb.prep.uaspeech",
    # Cross-pair
    "training.dataio.prep_aphasia_dbank":   "ahb.prep.cross_aphasia_dbank",
    "training.dataio.prep_c9s_avfad":       "ahb.prep.cross_c9s_avfad",
    "training.dataio.prep_c9s_coswara":     "ahb.prep.cross_c9s_coswara",
    "training.dataio.prep_coswara_avfad":   "ahb.prep.cross_coswara_avfad",
    "training.dataio.prep_edaic_iemocap":   "ahb.prep.cross_edaic_iemocap",
    "training.dataio.prep_edaic_ravdess":   "ahb.prep.cross_edaic_ravdess",
    "training.dataio.prep_mvdr_ksof":       "ahb.prep.cross_mvdr_ksof",
    "training.dataio.prep_mvdr_torgo":      "ahb.prep.cross_mvdr_torgo",
    "training.dataio.prep_mvdr_uaspeech":   "ahb.prep.cross_mvdr_uaspeech",
    "training.dataio.prep_ravdess_iemocap": "ahb.prep.cross_ravdess_iemocap",
    "training.dataio.prep_torgo_ksof":      "ahb.prep.cross_torgo_ksof",
    "training.dataio.prep_torgo_uaspeech":  "ahb.prep.cross_torgo_uaspeech",
    "training.dataio.prep_uaspeech_ksof":   "ahb.prep.cross_uaspeech_ksof",
    # Category-cross — all 6 legacy shims map to the single module that
    # exposes ``prepare_category`` plus the per-direction aliases.
    "training.dataio.prep_category_c1_c2":  "ahb.prep.category",
    "training.dataio.prep_category_c1_c3":  "ahb.prep.category",
    "training.dataio.prep_category_c1_c4":  "ahb.prep.category",
    "training.dataio.prep_category_c2_c3":  "ahb.prep.category",
    "training.dataio.prep_category_c2_c4":  "ahb.prep.category",
    "training.dataio.prep_category_c3_c4":  "ahb.prep.category",
    "training.dataio.prep_category_common": "ahb.prep.category",
}
