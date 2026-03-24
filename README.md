# Whisper + Visemas (Español)

Este repositorio ahora mantiene **dos pipelines compatibles**:

1. Pipeline original (ya existente):
- `whisper_to_json.py` (OpenAI Whisper) -> transcripción JSON con timestamps
- `build_ljspeech_dataset.py` -> dataset estilo LJSpeech/Piper

2. Pipeline nuevo:
- `main.py` + módulos `transcription.py`, `phonemizer_es.py`, `viseme_mapper.py`, `timing.py`, `export.py`, `utils.py`
- Genera tiempos de visemas para lip-sync de avatares/digital humans

Ambos conviven sin reemplazarse entre sí.

## Requisitos

- Python 3.10+
- `ffmpeg` en PATH (necesario para mp3/m4a y para extracción de segmentos)

Instalación:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Pipeline A (existente): Whisper -> JSON

```powershell
python whisper_to_json.py downloads\audio.mp3 -o salida.json -m large-v3 -l es
```

## Pipeline B (existente): JSON -> Dataset LJSpeech/Piper

```powershell
python build_ljspeech_dataset.py --input-json salida.json --output-dir .\dataset --sample-rate 22050 --metadata-format classic
```

## Pipeline C (nuevo): Audio español -> Visemas JSON

```powershell
python main.py --input input.wav --output visemes.json --mouth-cues mouth_cues.json
```

Defaults actuales del pipeline de visemas (optimizados para velocidad/recursos):
- `--model base`
- `--compute-type int8`
- `--beam-size 1`

Puedes subir calidad si necesitas:

```powershell
python main.py --input input.wav --output visemes.json --model small --beam-size 5 --compute-type auto
```

## Salidas del pipeline de visemas

- JSON principal con:
  - `metadata`
  - `audio_file`, `language`, `transcript`
  - `words` con visemas por palabra
  - `timeline` global de visemas (merge de repetidos adyacentes)

- JSON opcional de cues (`--mouth-cues`):

```json
{
  "mouthCues": [
    {"value": "OO", "start": 0.520, "end": 0.620},
    {"value": "TD", "start": 0.620, "end": 0.730}
  ]
}
```

## Notas

- El pipeline de visemas prioriza utilidad para animación (no investigación fonética).
- Labels compactos y estables para luego remapear a ARKit, Ready Player Me o controladores propios.
