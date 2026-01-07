# Known Issues

Current limitations in cruijff_kit with workarounds.

## Inspect Evals Don't Capture Model Names in Metadata

**Issue**: [#174](https://github.com/niznik-dev/cruijff_kit/issues/174)

Model names are not automatically recorded in inspect-ai evaluation metadata, making it harder to identify which model produced which results.

**Workaround**: Check the run directory name, which includes the model identifier.

## get_embeddings() Attention Mask Bug

**Issue**: [#234](https://github.com/niznik-dev/cruijff_kit/issues/234)

The `get_embeddings()` function has incorrect attention mask handling when processing batches with different sequence lengths.

**Workaround**: Use consistent batch sizes or process sequences of similar lengths together.

## Claude Code Agents Fail to Load

Sometimes Claude Code doesn't properly load the scaffold or run agents on startup.

**Workaround**: Restart Claude Code (`/quit` then relaunch).
