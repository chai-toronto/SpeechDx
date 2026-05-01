"""Category-cross prep for c3 (torgo/uaspeech/mvdr/ksof) <-> c4 (c9s/coswara/avfad).

Shim - all logic lives in ``prep_category_common.prepare_category``.
"""

from training.dataio.prep_category_common import prepare_category

prepare_category_c3_c4 = prepare_category
prepare_category_c4_c3 = prepare_category
