# Heads-up: this package shares its name with Python's standard-library `inspect`
# module — it is named after the external inspect-ai tool it wraps. No module in this
# subtree does a bare `import inspect`, and none should rely on one resolving to the
# stdlib: if this directory is ever placed on `sys.path` as a top-level entry, that
# import could bind here instead of to the stdlib. Keep stdlib introspection imports out
# of this subtree, or alias them explicitly.
