from pathlib import Path
from modelscope.hub.api import HubApi
import urllib.request
import json
import pandas as pd

root = Path('datasets/modelscope_speech_commands_small')
raw_dir = root / 'raw'
wav_dir = root / 'wavs'
raw_dir.mkdir(parents=True, exist_ok=True)
wav_dir.mkdir(parents=True, exist_ok=True)

api = HubApi()
namespace = 'DynamicSuperb'
dataset_name = 'SpeechCommandRecognition_GoogleSpeechCommandsV1'
file_path = 'data/test-00000-of-00001.parquet'
url = api.get_dataset_file_url(file_path, dataset_name, namespace)
parquet_path = raw_dir / 'test-00000-of-00001.parquet'
if not parquet_path.exists():
    print('downloading', url)
    urllib.request.urlretrieve(url, parquet_path)
else:
    print('parquet exists', parquet_path)

frame = pd.read_parquet(parquet_path)
metadata_path = root / 'metadata.jsonl'
written = 0
with metadata_path.open('w', encoding='utf-8') as meta:
    for _, row in frame.head(20).iterrows():
        audio = row.get('audio')
        if not isinstance(audio, dict) or not audio.get('bytes'):
            continue
        source_name = Path(audio.get('path') or row.get('file') or f'sample_{written:04d}.wav').name
        output_path = wav_dir / f'{written:04d}_{source_name}'
        output_path.write_bytes(audio['bytes'])
        record = {
            'index': written,
            'source_file': row.get('file'),
            'label': row.get('label'),
            'instruction': row.get('instruction'),
            'audio_path': str(output_path),
            'dataset_id': f'{namespace}/{dataset_name}',
        }
        meta.write(json.dumps(record, ensure_ascii=False) + '\n')
        written += 1
print('rows', len(frame))
print('written_wavs', written)
print('output_dir', wav_dir)
