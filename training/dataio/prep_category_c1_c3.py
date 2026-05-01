"""Category-cross prep for c1 (edaic/ravdess/iemocap) <-> c3 (torgo/uaspeech/mvdr/ksof).

Shim - all logic lives in ``prep_category_common.prepare_category``.
"""

from training.dataio.prep_category_common import prepare_category

prepare_category_c1_c3 = prepare_category
prepare_category_c3_c1 = prepare_category
