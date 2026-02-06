Do not change any code unless I specifically request it. By default, I'm looking for suggestions, both ideas on what to do and potential code snippets I could integrate.

When in doubt, do not include guards like `try: except`, `dictionary.get()`, `if len(data) == 0`, etc. I would much prefer if code obviously fails when something isn't as expected. I do not want silent failures or bloated code.

Do not add compatibility fallbacks for experiment-critical config or metadata. Required fields/files must be present and correct, and code should fail loudly when they are not.

Use a single explicit source of truth for layer selection. In experiment scripts, define `selected_layer_combination` directly and assert it exists in loaded AO/training config. Never infer layer selection from list order (for example: first/middle/last).

For multi-layer prompting and steering, assert formatting invariants. Layer blocks must have deterministic template/tokenization and identical placeholder-token counts per layer block.
