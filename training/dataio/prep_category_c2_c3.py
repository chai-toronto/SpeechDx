"""Category-cross prep for c2 (aphasia/dbank) <-> c3 (torgo/uaspeech/mvdr/ksof).

Shim - all logic lives in ``prep_category_common.prepare_category``.
"""

from training.dataio.prep_category_common import prepare_category

prepare_category_c2_c3 = prepare_category
prepare_category_c3_c2 = prepare_category
