"""Training CV keeps known speaker/source groups and augmentations together."""
import json
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from src.ml.training_config import SEED, ROOT


def original_group_folds(y, ids, n_splits=3):
    import pandas as pd
    train = pd.read_csv(ROOT / 'data/splits/train.csv', keep_default_na=False)
    if 'split_group_id' not in train:
        raise ValueError('Missing source groups. Prepare this dataset with the v5 kit.')
    lookup = dict(zip(train['audio_id'].astype(str), train['split_group_id'].astype(str)))
    ids = np.asarray(ids).astype(str); y = np.asarray(y)
    original_ids = [s.split('__aug')[0] for s in ids]
    if any(s not in lookup or not lookup[s] for s in original_ids):
        raise ValueError('Feature IDs do not match frozen source-group metadata.')
    groups = np.asarray([lookup[s] for s in original_ids])
    originals = np.asarray([i for i, s in enumerate(ids) if '__aug' not in s])
    if len(set(ids[originals])) != len(originals):
        raise ValueError('Duplicate original feature IDs.')
    for label in sorted(set(y)):
        if len(set(groups[y == label])) < n_splits:
            raise ValueError(f'{label}: need at least {n_splits} training source groups for CV.')
    # Feasibility is based on labels/groups only, never model scores.
    for seed in range(SEED, SEED + 64):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = []; evidence = []
        for _, held in cv.split(np.zeros((len(originals), 1)), y[originals], groups[originals]):
            held_idx = originals[held]; held_groups = set(groups[held_idx])
            fit_idx = np.asarray([i for i, g in enumerate(groups) if g not in held_groups])
            if set(y[fit_idx]) != set(y) or set(y[held_idx]) != set(y):
                break
            if set(groups[fit_idx]) & held_groups:
                raise RuntimeError('CV source group overlap.')
            folds.append((fit_idx, held_idx))
            evidence.append({'fit_samples_including_augmentation': len(fit_idx),
                             'held_originals': len(held_idx), 'held_groups': sorted(held_groups),
                             'held_class_counts': {str(c): int(np.sum(y[held_idx] == c)) for c in sorted(set(y))}})
        if len(folds) == n_splits:
            out = ROOT / 'reports/training_cv_groups.json'; out.parent.mkdir(exist_ok=True)
            out.write_text(json.dumps({'seed': seed, 'folds': evidence,
                'group_source': 'frozen split_group_id; user speaker groups plus historical recording groups',
                'source_independence_verified': False}, indent=2))
            return folds
    raise ValueError('Cannot make three CV folds with all classes. See source-group report.')
