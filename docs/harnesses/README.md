# Harness snapshots

The live archive is `runs/archive/` (gitignored, SQLite + a nested git repo).
These files are tracked snapshots of the verified best variants, so the result
survives a `runs/` cleanup.

## Best variant

`8f763e61.json` — hash
`8f763e61132dd09c448150c159dc39321a6e08dcb29c0437e960156ce87becd5`

| Split | Seed | Best | Delta |
|---|---:|---:|---:|
| Dev (14 tasks, repair) | 0.9048 | 1.0000 | +0.0952 |
| Hidden anchor (14 tasks, one-shot) | 0.8988 | 0.9345 | +0.0357 |

Lineage: seed `ef337c08` → `2b87a277` → `8f763e61`. A fourth proposal
`95e014bc` regressed dev to 0.9762 and was rejected.

Promote it in the live archive:

```bash
uv run janet archive promote 8f763e61132dd09c448150c159dc39321a6e08dcb29c0437e960156ce87becd5
uv run janet archive rollback
```

The hash covers `system_prompt`, `max_attempts`, and `samples_per_attempt`;
`name`, `parent`, and `mutation_note` are lineage metadata and are not part of
the hash.

## Caveat

The winning prompt names `flatten_nested`. The anchor holds out inputs, not
bugs, so this gain is real but narrow. The next phase adds held-out bug
variants before the loop is asked to demonstrate broad generalization.
