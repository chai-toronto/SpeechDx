"""Category-cross prep for c1 (edaic/ravdess/iemocap) <-> c2 (aphasia/dbank).

Shim - all logic lives in ``prep_category_common.prepare_category``.
"""

from training.dataio.prep_category_common import prepare_category

prepare_category_c1_c2 = prepare_category
prepare_category_c2_c1 = prepare_category
