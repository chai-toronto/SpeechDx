"""SpeechDx harness package.

Replaces the legacy ``run_all*.py`` scripts and the ``bench/`` orchestrator
package with a single entry point (``python -m sdx``) that decouples cache
warming from training: the encoder is loaded once per (task, encoder) pair
inside ``sdx.warm`` and dropped before any training process starts.
"""
