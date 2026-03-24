import os

from speechbrain.utils.train_logger import TrainLogger, TensorboardLogger


class CombinedLogger(TrainLogger):
    def __init__(self, save_dir):
        super().__init__()
        self.tensorboard = TensorboardLogger(save_dir)
        self.file_logger = SpreadsheetLogger(os.path.join(save_dir, "train_log.tsv"))

    def log_stats(
        self,
        *args,
        **kwargs
    ):
        self.tensorboard.log_stats(*args, **kwargs)
        self.file_logger.log_stats(*args, **kwargs)


class SpreadsheetLogger(TrainLogger):
    """Logger that outputs tab-separated values for easy spreadsheet pasting.

    Arguments
    ---------
    save_file : str
        The file to use for logging train information.
    precision : int
        Number of decimal places to display. Default 2.
    """

    def __init__(self, save_file, precision=2):
        self.save_file = save_file
        self.precision = precision
        self.header_written = os.path.exists(save_file)

    def _format_value(self, value):
        if isinstance(value, float) and 1.0 < value < 100.0:
            return f"{value:.{self.precision}f}"
        elif isinstance(value, float):
            return f"{value:.{self.precision}e}"
        return str(value)

    def log_stats(
        self,
        stats_meta,
        train_stats=None,
        valid_stats=None,
        test_stats=None,
        verbose=True,
    ):
        keys = []
        values = []

        for k, v in stats_meta.items():
            keys.append(k)
            values.append(self._format_value(v))

        for dataset, stats in [
            ("train", train_stats),
            ("valid", valid_stats),
            ("test", test_stats),
        ]:
            if stats is not None:
                for k, v in stats.items():
                    keys.append(f"{dataset}_{k}")
                    values.append(self._format_value(v))

        with open(self.save_file, "a", encoding="utf-8") as fout:
            if not self.header_written:
                print("\t".join(keys), file=fout)
                self.header_written = True
            print("\t".join(values), file=fout)

        if verbose:
            print("\t".join(values))
