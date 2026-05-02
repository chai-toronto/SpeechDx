# script/

## Google Drive sync agent (`sync_embeddings_to_drive.sh` + `com.lkieu.sync-embeddings.plist`)

Mirrors `embeddings_avg_final/` to Google Drive on every change. Local is
source of truth; Drive is a faithful mirror (`rsync -a --delete`).

### Architecture

- `sync_embeddings_to_drive.sh` — does an initial `rsync`, then watches the
  source dir with `fswatch -o` and re-syncs on every change event.
- `com.lkieu.sync-embeddings.plist` — launchd agent that runs the script at
  login and restarts it if it ever exits (`KeepAlive`).
- Logs: `~/Library/Logs/sync-embeddings.{out,err}.log`

### Install

```bash
# 1. fswatch
brew install fswatch

# 2. Symlink the plist into LaunchAgents (so edits in repo flow through)
ln -s "$PWD/com.lkieu.sync-embeddings.plist" \
      ~/Library/LaunchAgents/com.lkieu.sync-embeddings.plist

# 3. Grant Full Disk Access in System Settings -> Privacy & Security:
#    - Drag sync_embeddings_to_drive.sh into the FDA list, OR
#    - Add /bin/zsh as a fallback (broader but reliable)
#    Required because ~/Library/CloudStorage/ is TCC-protected.

# 4. Load the agent
launchctl bootstrap gui/$UID ~/Library/LaunchAgents/com.lkieu.sync-embeddings.plist
```

### Status

```bash
launchctl list | grep sync-embeddings    # PID = alive, exit code 0 = healthy
tail -f ~/Library/Logs/sync-embeddings.err.log
```

Smoke test:
```bash
touch /Users/lkieu/PycharmProjects/Audio-Health-Benchmark/embeddings_avg_final/.sync-test
sleep 3
ls "$HOME/Library/CloudStorage/GoogleDrive-tiendat691@gmail.com/My Drive/health datasets/embeddings_avg_final/.sync-test"
rm  /Users/lkieu/PycharmProjects/Audio-Health-Benchmark/embeddings_avg_final/.sync-test
```

### Reload after editing script or plist

```bash
launchctl bootout   gui/$UID/com.lkieu.sync-embeddings
launchctl bootstrap gui/$UID ~/Library/LaunchAgents/com.lkieu.sync-embeddings.plist
```

### Uninstall

```bash
launchctl bootout gui/$UID/com.lkieu.sync-embeddings
rm ~/Library/LaunchAgents/com.lkieu.sync-embeddings.plist
# Optional: also delete script + plist from this folder, the log files,
# and the FDA grant in System Settings.
```

### Gotchas

- **TCC is evaluated at process start.** After granting FDA, the running
  agent still has the old permissions — bootout/bootstrap to pick them up.
- **FDA on shell scripts is finicky.** The Settings file picker grays out
  `.sh` files; drag from Finder, or grant FDA to `/bin/zsh` instead.
- **rsync to Drive is two-stage.** rsync writes to Drive Desktop's local
  cache (fast); Drive Desktop uploads to Google in the background (network-
  bound). The menu-bar Drive icon is the truth about real upload progress.
- **rsync -a is silent.** Add `--info=progress2` if you want a live counter
  when running manually.
- **Absolute paths.** Both the script and plist hardcode the repo path. If
  you move the repo, edit them.
