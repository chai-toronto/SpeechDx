from training.brain import DiagnosticsBrain


class CVBrain:
    """Wrapper to train multiple folds simultaneously and report average metrics."""

    def __init__(self, hparams, run_opts, num_folds):
        self.hparams = hparams
        self.run_opts = run_opts
        self.num_folds = num_folds
        self.brains = []

        # Create a brain instance for each fold
        for fold in range(num_folds):
            brain = DiagnosticsBrain(
                modules=hparams["modules"],
                opt_class=hparams["opt_class"],
                hparams=hparams,
                run_opts=run_opts,
                checkpointer=hparams[f"checkpointer_fold_{fold}"] if f"checkpointer_fold_{fold}" in hparams else
                hparams["checkpointer"],
            )
            self.brains.append(brain)

