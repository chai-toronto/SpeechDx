"""Manifest preparation modules — one per dataset and per cross-pair.

Each module exposes one or more ``prepare_*`` functions invoked by
``sdx.prep.dispatch.ensure_manifest``. Old task YAMLs reference the legacy
``training.dataio.prep_*`` paths; ``dispatch`` maps those to the modules
in this package via :data:`OLD_TO_NEW_MODULE`.
"""

from __future__ import annotations

OLD_TO_NEW_MODULE = {
    # Single-dataset
    "training.dataio.prep_default":         "sdx.prep.standard",
    "training.dataio.prep_aphasia":         "sdx.prep.aphasia",
    "training.dataio.prep_avfad":           "sdx.prep.avfad",
    "training.dataio.prep_c19sounds":             "sdx.prep.c19sounds",
    "training.dataio.prep_coswara":         "sdx.prep.coswara",
    "training.dataio.prep_dementiabank":           "sdx.prep.dementiabank",
    "training.dataio.prep_edaic":           "sdx.prep.edaic",
    "training.dataio.prep_iemocap":         "sdx.prep.iemocap",
    "training.dataio.prep_ksof":            "sdx.prep.ksof",
    "training.dataio.prep_mdvr":            "sdx.prep.mdvr",
    "training.dataio.prep_nemours":         "sdx.prep.nemours",
    "training.dataio.prep_ravdess":         "sdx.prep.ravdess",
    "training.dataio.prep_torgo":           "sdx.prep.torgo",
    "training.dataio.prep_uaspeech":        "sdx.prep.uaspeech",
    # Cross-pair
    "training.dataio.prep_aphasia_dementiabank":   "sdx.prep.cross_aphasia_dementiabank",
    "training.dataio.prep_c19sounds_avfad":       "sdx.prep.cross_c19sounds_avfad",
    "training.dataio.prep_c19sounds_coswara":     "sdx.prep.cross_c19sounds_coswara",
    "training.dataio.prep_coswara_avfad":   "sdx.prep.cross_coswara_avfad",
    "training.dataio.prep_edaic_iemocap":   "sdx.prep.cross_edaic_iemocap",
    "training.dataio.prep_edaic_ravdess":   "sdx.prep.cross_edaic_ravdess",
    "training.dataio.prep_mdvr_ksof":       "sdx.prep.cross_mdvr_ksof",
    "training.dataio.prep_mdvr_torgo":      "sdx.prep.cross_mdvr_torgo",
    "training.dataio.prep_mdvr_uaspeech":   "sdx.prep.cross_mdvr_uaspeech",
    "training.dataio.prep_ravdess_iemocap": "sdx.prep.cross_ravdess_iemocap",
    "training.dataio.prep_torgo_ksof":      "sdx.prep.cross_torgo_ksof",
    "training.dataio.prep_torgo_uaspeech":  "sdx.prep.cross_torgo_uaspeech",
    "training.dataio.prep_uaspeech_ksof":   "sdx.prep.cross_uaspeech_ksof",
    # Category-cross — all 6 legacy shims map to the single module that
    # exposes ``prepare_category`` plus the per-direction aliases.
    "training.dataio.prep_category_c1_c2":  "sdx.prep.category",
    "training.dataio.prep_category_c1_c3":  "sdx.prep.category",
    "training.dataio.prep_category_c1_c4":  "sdx.prep.category",
    "training.dataio.prep_category_c2_c3":  "sdx.prep.category",
    "training.dataio.prep_category_c2_c4":  "sdx.prep.category",
    "training.dataio.prep_category_c3_c4":  "sdx.prep.category",
    "training.dataio.prep_category_common": "sdx.prep.category",
}
