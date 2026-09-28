"""Decode, audit and extend historical splits without editing incoming recordings."""
from pathlib import Path
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import csv
import hashlib
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import soundfile as sf
from src.audio.decode import decode_mono, SAMPLE_RATE, ffmpeg_executable
from src.ml.training_config import CLASSES, FOLDER_TO_LABEL, LABEL_TO_FOLDER

SPLITS = ('train', 'validation', 'test')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            # Stable representatives, independent of worker completion order.
            self.parent[max(a, b)] = min(a, b)


def history_tables(root):
    records = defaultdict(list)
    for name in ['reviewed9_metadata.csv', 'previous_metadata.csv']:
        if (root / 'history' / name).is_file():
            for r in read_csv(root / 'history' / name):
                if r.get('sha256'):
                    records[r['sha256']].append(r)
    old_holds = {}
    for r in read_csv(root / 'history/previous_audit.csv'):
        if r['status'] in {'LABEL_CONFLICT', 'HISTORICAL_LABEL_CHANGED', 'REVIEW_EXCLUDED'}:
            old_holds[r['sha256']] = r['status']
    decisions = {}
    for name in ['reviewed_labels.json', 'aggression_decisions.json']:
        payload = json.loads((root / 'history' / name).read_text())
        for d in payload['decisions']:
            if not (d.get('reviewed') or d.get('reviewer_listened')):
                continue
            decisions[d['sha256']] = {
                'action': 'keep' if d['action'] == 'keep_aggression' else d['action'], 'label': d.get('target_label') or d.get('label'),
                'notes': d.get('notes', '')}
    speakers = {r['sha256']: r['group_id'] for r in json.loads((root / 'history/speaker_groups.json').read_text())}
    return records, old_holds, decisions, speakers


def assign_groups(rows):
    """Deterministic class-balanced 70/15/15 target with indivisible groups.

    No audio features, fitted predictions or test scores enter this procedure.
    Historical assignments take precedence. Unknown new Aggression speakers
    remain train-only; their identity is not represented as verified.
    """
    totals = Counter(r['class_label'] for r in rows)
    target = np.array([[totals[c] * .70, totals[c] * .15, totals[c] * .15] for c in CLASSES])
    groups = defaultdict(list)
    for r in rows:
        groups[r['split_group_id']].append(r)
    counts = np.zeros((len(CLASSES), 3), dtype=int)
    assigned = {}; vectors = {}; forced = set()
    for group, members in groups.items():
        vector = np.array([sum(r['class_label'] == c for r in members) for c in CLASSES])
        vectors[group] = vector
        historical = {r['historical_split'] for r in members if r['historical_split']}
        if len(historical) > 1:
            raise ValueError('Historical source group conflict was not held: ' + group)
        split = next(iter(historical), None)
        if split is None and any(r['class_label'] == 'Aggression' and not r['speaker_group_id'] for r in members):
            split = 'train'
        if split:
            index = SPLITS.index(split); assigned[group] = index
            counts[:, index] += vector; forced.add(group)
    scale = np.maximum(target.sum(axis=1), 1)[:, None]

    def objective(current):
        return float(np.sum((current - target) ** 2 / scale) +
                     .2 * np.sum((current.sum(axis=0) - target.sum(axis=0)) ** 2) / max(len(rows), 1))

    free = sorted((g for g in groups if g not in assigned),
                  key=lambda g: (-int(vectors[g].sum()), hashlib.sha256(g.encode()).hexdigest()))
    for group in free:
        options = []
        for index in range(3):
            candidate = counts.copy(); candidate[:, index] += vectors[group]
            options.append((objective(candidate), index))
        index = min(options)[1]; assigned[group] = index; counts[:, index] += vectors[group]
    # Whole-group moves improve rounding without ever splitting a source.
    for _ in range(8):
        moved = False
        for group in free:
            old = assigned[group]; best = (objective(counts), old)
            for index in range(3):
                candidate = counts.copy(); candidate[:, old] -= vectors[group]; candidate[:, index] += vectors[group]
                if np.any(candidate <= 0):
                    continue
                value = objective(candidate)
                if value < best[0] - 1e-9:
                    best = (value, index)
            if best[1] != old:
                counts[:, old] -= vectors[group]; counts[:, best[1]] += vectors[group]
                assigned[group] = best[1]; moved = True
        if not moved:
            break
    for r in rows:
        r['split'] = SPLITS[assigned[r['split_group_id']]]
    for c, values in zip(CLASSES, counts):
        if np.any(values == 0):
            raise ValueError(f'{c}: no examples in one split after preserving whole groups. See preparation audit.')
    return {s: {c: int(counts[i, j]) for i, c in enumerate(CLASSES)} for j, s in enumerate(SPLITS)}


def prepare(root=ROOT, workers=2):
    root = Path(root)
    if (root / 'COPY_COMPLETE.json').exists():
        from scripts.verify_prepared import verify
        verify(root)
        old = json.loads((root / 'COPY_COMPLETE.json').read_text())
        if old['incoming_manifest_sha256'] != sha(root / 'incoming_manifest.csv'):
            raise ValueError('Incoming dataset changed. Use a new run directory.')
        print('PREPARATION ALREADY COMPLETE. Existing splits retained.', flush=True)
        return old
    (root / 'COPY_INCOMPLETE').write_text('Preparation in progress. Training is blocked.\n')
    report = root / 'reports/dataset_preparation'; report.mkdir(parents=True, exist_ok=True)
    manifest = read_csv(root / 'incoming_manifest.csv')
    histories, old_holds, decisions, known_speakers = history_tables(root)
    cache = root / '.preparation_cache'; cache.mkdir(exist_ok=True)
    files = defaultdict(list)
    for row in manifest:
        path = (root / 'incoming' / row['relative_path']).resolve()
        if not path.is_relative_to((root / 'incoming/data/raw').resolve()) or not path.is_file():
            raise ValueError('Missing or unsafe incoming path: ' + row['relative_path'])
        if sha(path) != row['sha256']:
            raise ValueError('Input hash changed: ' + row['relative_path'])
        if row['folder'] not in FOLDER_TO_LABEL:
            raise ValueError('Unknown folder: ' + row['folder'])
        files[row['sha256']].append(row)
    print(f'RAW: {len(manifest)} | Unique file bytes: {len(files)} | Decoder: {ffmpeg_executable()}', flush=True)

    def decode(item):
        digest, aliases = item
        entry = {'source_sha256': digest}
        try:
            y, sr = decode_mono(root / 'incoming' / aliases[0]['relative_path'])
            duration = len(y) / sr
            entry.update(duration=duration, sample_rate=sr, channels=1,
                         peak=float(np.max(np.abs(y))), rms=float(np.sqrt(np.mean(y.astype('float64') ** 2))))
            entry['pcm_sha256'] = hashlib.sha256(b'22050/mono/f32le\0' + y.astype('<f4').tobytes()).hexdigest()
            if duration < .30:
                entry['decode_status'] = 'TOO_SHORT'
            elif duration > 60.001:
                entry['decode_status'] = 'TOO_LONG'
            elif entry['peak'] < 1e-7:
                entry['decode_status'] = 'SILENT'
            else:
                entry['decode_status'] = 'OK'
                sf.write(cache / (digest + '.wav'), y, sr, subtype='FLOAT')
        except Exception as exc:
            entry.update(decode_status='DECODE_FAILED', error=str(exc))
        return digest, entry

    decoded = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for number, (digest, result) in enumerate(pool.map(decode, sorted(files.items())), 1):
            decoded[digest] = result
            if number % 100 == 0 or number == len(files):
                print(f'Decoded {number}/{len(files)}', flush=True)
    pcm_groups = defaultdict(list)
    for digest, entry in decoded.items():
        pcm_groups[entry.get('pcm_sha256') or ('bytes:' + digest)].append(digest)

    audit = []; retained = []; union = UnionFind(); history_by_token = defaultdict(set)
    for pcm, hashes in sorted(pcm_groups.items()):
        aliases = [r for h in hashes for r in files[h]]
        prior = [r for h in hashes for r in histories.get(h, [])]
        human = [decisions[h] for h in hashes if h in decisions]
        labels = {FOLDER_TO_LABEL[r['folder']] for r in aliases}
        old_labels = {r['class_label'] for r in prior}
        old_splits = {r['split'] for r in prior if r.get('split') in SPLITS}
        group_keys = set()
        for r in prior:
            group_keys.add('history:' + (r.get('original_audio_id') or r['audio_id']))
        speaker_ids = set()
        for r in aliases:
            speaker = known_speakers.get(r['sha256'])
            if not speaker and r.get('speaker_folder'):
                speaker = 'FOLDER:' + r['speaker_folder']
            if speaker:
                speaker_ids.add(speaker); group_keys.add('speaker:' + speaker)
        content_id = 'PCM-' + pcm
        group_keys.add(content_id)
        for key in group_keys:
            union.union(content_id, key)
            history_by_token[key].update(old_splits)
        status = 'OK'; reason = ''; label = next(iter(sorted(labels)))
        reviewed = False
        if any(d['action'] in {'exclude', 'hold'} for d in human):
            status = 'REVIEW_EXCLUDED'; reason = 'Human review holds/excludes this exact audio.'
        elif human:
            approved = {d['label'] or label for d in human if d['action'] in {'keep', 'relabel'}}
            if len(approved) == 1:
                label = next(iter(approved)); reviewed = True
            elif approved:
                status = 'REVIEW_CONFLICT'; reason = 'Human decisions disagree for identical decoded content.'
        if status == 'OK' and not reviewed:
            if len(labels) > 1:
                status = 'LABEL_CONFLICT'; reason = 'Identical decoded audio occurs under multiple class labels.'
            elif any(h in old_holds for h in hashes):
                status = 'PRIOR_REVIEW_HOLD'; reason = 'A historical label conflict or exclusion still requires a human decision.'
        if status == 'OK' and old_labels and old_labels != {label}:
            # Never change held-out labels as a side effect of preparation.
            if old_splits & {'validation', 'test'} or not reviewed:
                status = 'HISTORICAL_LABEL_CHANGED'; reason = 'Current label differs from historical label; held out from this version.'
        if status == 'OK' and len(old_splits) > 1:
            status = 'HISTORICAL_SPLIT_CONFLICT'; reason = 'This content previously occurred in multiple splits.'
        if label not in CLASSES:
            status = 'UNKNOWN_REVIEW_LABEL'; reason = 'Human target label is outside the ten-class schema.'
        # Prefer a known historical recording and a known speaker path, then a stable path.
        aliases.sort(key=lambda r: (r['sha256'] not in histories, not bool(r.get('speaker_folder')), r['relative_path']))
        owner = aliases[0]; technical = decoded[owner['sha256']]
        if status == 'OK' and technical['decode_status'] != 'OK':
            status = technical['decode_status']; reason = technical.get('error', '')
        historical = next(iter(old_splits), '') if len(old_splits) <= 1 else ''
        group_status = 'user_assigned_speaker_group' if speaker_ids else 'unverified_recording_hash_fallback'
        entry = {'audio_id': 'AUD-' + owner['sha256'], 'original_audio_id': content_id,
                 'source_relative_path': owner['relative_path'], 'source_sha256': owner['sha256'],
                 'class_label': label, 'status': status, 'pcm_sha256': technical.get('pcm_sha256', ''),
                 'duration': technical.get('duration', ''), 'sample_rate': SAMPLE_RATE, 'channels': 1,
                 'peak': technical.get('peak', ''), 'rms': technical.get('rms', ''),
                 'speaker_group_id': '|'.join(sorted(speaker_ids)), 'source_group_status': group_status,
                 'historical_split': historical, 'history': 'previously_used_' + historical if historical else 'new_to_recorded_metadata',
                 'label_review_status': 'human_reviewed' if reviewed else 'folder_label_unverified',
                 'label_review_notes': ' | '.join(d['notes'] for d in human),
                 'original_or_augmented': 'original', 'error': reason, '_content_token': content_id}
        if status == 'OK':
            retained.append(entry)
        for index, alias in enumerate(aliases):
            audit.append({**{k: v for k, v in entry.items() if not k.startswith('_')},
                          'source_relative_path': alias['relative_path'], 'source_sha256': alias['sha256'],
                          'original_folder_label': FOLDER_TO_LABEL[alias['folder']],
                          'status': ('DUPLICATE' if status == 'OK' and index else status),
                          'duplicate_of': owner['relative_path'] if index else ''})
    # Connect history assignments even when a conflicting alias was excluded above.
    forced_by_root = defaultdict(set)
    for token, splits in history_by_token.items():
        forced_by_root[union.find(token)].update(splits)
    retained_ids = set()
    clean = []
    for row in retained:
        token = union.find(row.pop('_content_token'))
        frozen = forced_by_root[token]
        row['split_group_id'] = 'GROUP-' + hashlib.sha256(token.encode()).hexdigest()
        if len(frozen) > 1:
            for a in audit:
                if a['audio_id'] == row['audio_id']:
                    a['status'] = 'SOURCE_SPLIT_CONFLICT'; a['error'] = 'Whole source/speaker connects historical splits; excluded, never moved.'
            continue
        if frozen:
            row['historical_split'] = next(iter(frozen))
            row['history'] = 'previously_used_' + row['historical_split']
        clean.append(row); retained_ids.add(row['audio_id'])
    write_csv(report / 'audio_audit.csv', audit)
    if not clean:
        raise ValueError('No eligible audio remained. See reports/dataset_preparation/audio_audit.csv.')
    split_counts = assign_groups(clean)
    for number, row in enumerate(clean, 1):
        source = cache / (row['source_sha256'] + '.wav')
        relative = f"data/raw/{LABEL_TO_FOLDER[row['class_label']]}/{row['audio_id']}.wav"
        destination = root / relative; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        row.update(relative_path=relative, filename=destination.name, sha256=sha(destination), file_size_bytes=destination.stat().st_size)
        if number % 250 == 0 or number == len(clean):
            print(f'Prepared {number}/{len(clean)}', flush=True)
    clean.sort(key=lambda r: (r['class_label'], r['audio_id']))
    write_csv(root / 'data/metadata.csv', clean)
    for split in SPLITS:
        write_csv(root / f'data/splits/{split}.csv', [r for r in clean if r['split'] == split])
    counts = Counter(r['class_label'] for r in clean)
    used_hashes = {a['source_sha256'] for a in audit if a['audio_id'] in retained_ids}
    missing = []
    for h, prior in histories.items():
        if h not in used_hashes and any(r.get('split') in {'validation', 'test'} for r in prior):
            missing.append({'sha256': h, 'prior_labels': '|'.join(sorted({r['class_label'] for r in prior})),
                            'prior_splits': '|'.join(sorted({r.get('split', '') for r in prior})),
                            'reason': 'not present, rejected or held in this dataset version'})
    write_csv(report / 'historical_heldout_missing_or_held.csv', missing)
    write_csv(report / 'speaker_and_source_groups.csv', [
        {k: r[k] for k in ['audio_id', 'class_label', 'split', 'split_group_id', 'speaker_group_id', 'source_group_status']}
        for r in clean])
    actual = Counter(r['split'] for r in clean)
    summary = {
        'version': 'sonic-submission-v5', 'created_at': datetime.now(timezone.utc).isoformat(),
        'raw_files': len(manifest), 'retained_unique_files': len(clean),
        'status_counts': dict(Counter(r['status'] for r in audit)),
        'usable_counts': {c: counts[c] for c in CLASSES}, 'split_class_counts': split_counts,
        'class_imbalance_ratio_largest_to_smallest': round(max(counts.values()) / min(counts[c] for c in CLASSES), 4),
        'split_counts': {s: actual[s] for s in SPLITS},
        'requested_split_ratios': {'train': .7, 'validation': .15, 'test': .15},
        'actual_split_percentages': {s: round(100 * actual[s] / len(clean), 3) for s in SPLITS},
        'all_ten_300_file_target': all(counts[c] >= 300 for c in CLASSES),
        'at_least_3000_files': len(clean) >= 3000,
        'source_independence_verified': False,
        'user_assigned_speaker_groups': len({r['speaker_group_id'] for r in clean if r['speaker_group_id']}),
        'labels_human_reviewed': sum(r['label_review_status'] == 'human_reviewed' for r in clean),
        'historical_heldout_missing_or_held': len(missing),
        'metadata_sha256': sha(root / 'data/metadata.csv'),
        'incoming_manifest_sha256': sha(root / 'incoming_manifest.csv'),
        'prepared_split_csv_sha256': {s: sha(root / f'data/splits/{s}.csv') for s in SPLITS},
        'notes': [
            '70/15/15 is the target; indivisible speaker/source groups and historical membership can prevent exact class percentages.',
            'New Aggression recordings without a supplied speaker group are train-only. Unknown speakers may still overlap known speakers; full independence is not proved.',
            'Historical test recordings have already been evaluated. Changed test membership is explicitly reported; old and new overall accuracy are not directly comparable.',
            'Byte and canonical decoded-PCM equality remove exact duplicates, not all crops, re-encodings or shared sessions.',
            'Duration >60 seconds is held for event segmentation; silence/short/failed/conflicting files are held, not deleted.',
            'Most class labels come from supplied folders and are not listening-verified. No trained weights were copied.'
        ]}
    (report / 'summary.json').write_text(json.dumps(summary, indent=2))
    (root / 'COPY_COMPLETE.json').write_text(json.dumps(summary, indent=2))
    (root / 'COPY_INCOMPLETE').unlink()
    from scripts.verify_prepared import verify
    verify(root)
    print(json.dumps(summary, indent=2), flush=True)
    print('PREPARATION COMPLETE. No source recordings were deleted or relabelled on disk.', flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise SystemExit('Use --workers between 1 and 8.')
    prepare(workers=args.workers)
