"""Category-cross prep for c1 (edaic/ravdess/iemocap) <-> c4 (c9s/coswara/avfad).

Shim - all logic lives in ``prep_category_common.prepare_category``.
"""

from training.dataio.prep_category_common import prepare_category

prepare_category_c1_c4 = prepare_category
prepare_category_c4_c1 = prepare_category
