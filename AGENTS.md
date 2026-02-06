Do not change any code unless I specifically request it. By default, I'm looking for suggestions, both ideas on what to do and potential code snippets I could integrate.

When in doubt, do not include guards like `try: except`, `dictionary.get()`, `if len(data) == 0`, etc. I would much prefer if code obviously fails when something isn't as expected. I do not want silent failures or bloated code.

Avoid implicit assumptions about layer selection. Do not pick a layer by index or "middle layer" heuristics unless explicitly required by the experiment. In eval scripts, define an explicit `selected_layer_combination` constant and assert that it exists in the loaded AO/training config before applying settings.

Do not make silent global choices that materially change experiment semantics. If a code path can select among multiple valid configurations (for example: layer combinations, dataset mixtures, evaluation subsets, prompt templates, or steering layouts), require the selection to be explicit in the experiment file and assert it matches loaded config/metadata. Never default to "first", "middle", or similar implicit ordering unless the experiment explicitly defines that policy.
