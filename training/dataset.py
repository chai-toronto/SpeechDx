import speechbrain as sb

class CachableDataset(sb.dataio.dataset.DynamicItemDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = {}