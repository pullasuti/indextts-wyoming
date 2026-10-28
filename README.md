# IndexTTS Wyoming bridge

Streaming text-to-speech for Home Assistant, using
[IndexTTS-2.5](https://github.com/index-tts/index-tts) with zero-shot voice
cloning and emotion control. The model runs in-process and speaks the
[Wyoming protocol](https://github.com/OHF-Voice/wyoming), so HA connects to it
directly.

## Quick start

```bash
./run.sh
```

The first run installs uv if needed, clones index-tts at a pinned commit into
`repo/`, builds its Python 3.11 venv (`uv sync`), and downloads
`IndexTeam/IndexTTS-2.5` plus its auxiliary models (~10 GB). Later runs start
straight away.

The bridge listens on `tcp://0.0.0.0:10300` and accepts connections while the
model is still loading (~15 s, plus ~6 s of warmup).

- `SETUP_ONLY=1 ./run.sh` sets everything up without starting the bridge.
- `REINSTALL=1 ./run.sh` reinstalls the Python dependencies.
- Extra arguments go to `bridge_indextts.py` (`./run.sh --help` lists them).

Requires Linux with an NVIDIA GPU and ~6 GB of free VRAM.

## Configuration

Copy `.env.example` to `.env` and edit. Everything is optional:

| Variable             | Default                  | Description                                          |
|----------------------|--------------------------|------------------------------------------------------|
| `PROMPT_WAV`         | `voices/nicoleheart.wav` | Reference voice (10-15 s clean mono WAV)             |
| `VOICE`              | `nicoleheart`            | Voice name shown in Home Assistant                   |
| `DEFAULT_MOOD`       | `warm`                   | Mood for text before any mood tag (empty = the reference clip's own emotion) |
| `EMO_STRENGTH`       | `1.0`                    | Multiplier on every mood's emotion vector            |
| `URI`                | `tcp://0.0.0.0:10300`    | Wyoming listen address                               |
| `LANG_CODE`          | `EN`                     | Model language (EN, ZH, JA, ES, AR)                  |
| `SPEED`              | `1.0`                    | Speaking rate                                        |
| `NUM_BEAMS`          | `1`                      | GPT beam width (model default 3, ~0.2 s slower)      |
| `PREBUFFER_SECONDS`  | `1`                      | Audio held before playback starts                    |
| `MIN_MERGED_SECONDS` | `2.5`                    | Speech merged per model call after the first sentence |
| `EXTRA_ARGS`         |                          | Extra bridge arguments, string or bash array         |

## Tags

Mood tags set an 8-dimensional emotion vector (happy, angry, sad, afraid,
disgusted, melancholic, surprised, calm). The weights per mood are in `MOODS`
at the top of `bridge_indextts.py`. A mood stays until the next mood tag; a
sentence with a mood tag starts a new model call, so it doesn't colour the
sentences before it.

- Moods: `[warm] [confident] [happy] [excited] [serious] [calm] [sad]
  [melancholic] [angry] [afraid] [surprised] [disgusted]`
- Delivery: `[slow] [fast]`

`ha_prompt.txt` is a matching system-prompt section for the HA conversation
agent: every tag it teaches is one the bridge voices.

IndexTTS has no inline sounds. `[laughter] [sigh] [breath] [cough] [noise]
[lipsmack] [mn]` and `[whisper] [loud] [robot]` are removed (logged once), so
a prompt written for the CosyVoice bridge works unchanged.

## Latency

IndexTTS returns audio one text segment at a time, so the first audio arrives
when the whole opening sentence is generated: ~1.7-2.4 s on an RTX 4070
(RTF ~0.7-0.9 per sentence). Every model call costs ~1.3 s of fixed overhead,
so a very short opening sentence ("Hello!") can leave a pause of up to ~1 s
before the next one.

## systemd

`wyoming-indextts.service` runs `run.sh`. Install it with:

```bash
cp wyoming-indextts.service /etc/systemd/system/
systemctl daemon-reload
systemctl start wyoming-indextts
```

It declares `Conflicts=fish-wyoming.service` because both need most of a
12 GB GPU: starting either stops the other. Keep it disabled at boot if
another TTS unit is enabled.

## Testing

With the bridge running:

```bash
repo/.venv/bin/python test_bridge.py --port 10300 \
    --tagged "[happy] Hello! [laughter] This is a test."
```

It mimics HA: a cold first request, a one-shot, a stream with the compat
`synthesize` HA sends alongside it, a late duplicate (should time out), and a
blank one-shot (an empty audio-start/audio-stop pair).

## License

IndexTTS-2.5 code and weights are under the bilibili Model Use License
Agreement (see `repo/LICENSE`); they are downloaded at setup, not included
here.
