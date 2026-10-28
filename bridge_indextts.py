#!/usr/bin/env python3
"""IndexTTS-2.5 Wyoming TTS bridge (in-process).

Same pipeline as the CosyVoice bridge, with IndexTTS-2.5 as the engine:

  1. SENTENCE MERGING — the opening sentence goes alone (latency first);
     later sentences are batched to --min-merged-seconds of estimated speech.
  2. PRE-BUFFERING (jitter buffer) — AudioStart deferred until
     --prebuffer-seconds is queued; afterwards every chunk is written
     immediately.
  3. UNIFORM RE-CHUNKING — PCM is re-emitted in fixed --samples-per-chunk
     AudioChunks, remainder flushed once at the end of the utterance.
  4. WYOMING ERROR EVENTS — a failed block is reported with
     wyoming.error.Error; the remaining blocks still run.
  5. GRACEFUL SHUTDOWN — SIGINT/SIGTERM cancel the server cleanly.
  6. ONE-SHOT REQUESTS — a plain synthesize (tts.speak, "Try voice") is
     answered with audio-start/chunks/audio-stop. The copy HA sends
     alongside a stream, or right after it, is ignored.
  7. WARMUP — throwaway syntheses after load, so the first real request
     starts as fast as later ones (the cold first call is ~2x slower).

IndexTTS-specific:
  * Moods are 8-dim emotion vectors (happy, angry, sad, afraid, disgusted,
    melancholic, surprised, calm), see MOODS. The mood is sticky until the
    next mood tag and applies to the whole merged block.
  * [slow]/[fast] change duration_factor.
  * The model has NO inline sound tags. [laughter], [sigh], ... and the
    delivery tags it can't do ([whisper], [loud], [robot]) are stripped, so
    a prompt written for another bridge still works. Each is logged once.
  * Audio arrives one text segment at a time (the model has no finer
    streaming), so time to first audio = generation time of the opening
    sentence.

Startup order: bind the port immediately, load the model in the background,
queue any sentences that arrive during the load.

Must run inside the IndexTTS venv (repo/.venv) with wyoming installed.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import signal
import sys
import threading
import time
from functools import partial
from pathlib import Path
from typing import Any, Iterator, Optional

import torch

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
)

_LOGGER = logging.getLogger(__name__)

SAMPLE_RATE = 22050

# Rough speech-rate estimate for merging decisions: ~2.8 words/second.
WORDS_PER_SECOND = 2.8

# A one-shot synthesize arriving this soon after synthesize-stop, with
# matching text, is a late duplicate of the stream just played, not a request.
LATE_DUPLICATE_WINDOW_S = 3.0

# Synthesized once per warmup mood at startup (cold first call ~6 s vs ~2.5 s).
WARMUP_TEXT = "Hello there, this is a warm up."

# ---------------------------------------------------------------------------
# Tag handling
# ---------------------------------------------------------------------------

EMO_DIMS = ("happy", "angry", "sad", "afraid", "disgusted",
            "melancholic", "surprised", "calm")

# Mood -> emotion weights (missing dims are 0). The model's own code caps
# nothing on this path, so keep each mood's sum <= ~0.8. Starting values,
# tune by ear: [angry] at 0.8 sounded over the top in the smoke test.
MOODS: dict[str, dict[str, float]] = {
    "warm": {"happy": 0.3, "calm": 0.3},
    "confident": {"calm": 0.45, "happy": 0.15},
    "happy": {"happy": 0.6},
    "excited": {"happy": 0.5, "surprised": 0.3},
    "serious": {"calm": 0.55},
    "calm": {"calm": 0.6},
    "sad": {"sad": 0.55, "melancholic": 0.15},
    "melancholic": {"melancholic": 0.6},
    "angry": {"angry": 0.4},
    "afraid": {"afraid": 0.5},
    "surprised": {"surprised": 0.55},
    "disgusted": {"disgusted": 0.5},
}

# Delivery tags the model can do, as duration_factor multipliers (>1 = slower).
DELIVERY = {"slow": 1.2, "fast": 0.85}

# Tags other bridges' prompts use that this model can't voice: stripped.
UNSUPPORTED = {
    "laughter", "breath", "sigh", "cough", "noise", "lipsmack", "mn",
    "whisper", "loud", "robot",
}

ALIASES = {
    "confidence": "confident",
    "excitement": "excited",
    "cheerful": "happy",
    "annoyed": "angry",
    "scared": "afraid",
    "fearful": "afraid",
    "melancholy": "melancholic",
    "laughing": "laughter",
    "laughs": "laughter",
    "laugh": "laughter",
    "breaths": "breath",
    "sighs": "sigh",
    "coughs": "cough",
}

TAG_RE = re.compile(r"[\[\{]([^\]\}]+)[\]\}]")
PAREN_RE = re.compile(r"\(\s*([a-zA-Z]+)\s*\)")

_TERM_RE = re.compile(r"[.!?…][\"'”’)\]]*\s+")
_WORD_RE = re.compile(r"[A-Za-z.]+$")

_ABBREVIATIONS = {
    "e.g", "i.e", "eg", "ie", "etc", "mr", "mrs", "ms", "dr", "prof",
    "sr", "jr", "vs", "st",
}

_warned_tags: set[str] = set()


def _known(key: str) -> bool:
    return key in MOODS or key in DELIVERY or key in UNSUPPORTED


def normalize(text: str) -> str:
    """Turn '(laughs)'-style tags into '[laughter]'; leave other parens."""
    def fix(m: re.Match) -> str:
        key = m.group(1).lower()
        key = ALIASES.get(key, key)
        return f"[{key}]" if _known(key) else m.group(0)
    return PAREN_RE.sub(fix, text)


def parse_tags(text: str) -> tuple[str, Optional[str], Optional[str]]:
    """Split text into (spoken text, last mood or None, last delivery or None)."""
    text = normalize(text)
    mood: Optional[str] = None
    delivery: Optional[str] = None
    out: list[str] = []
    pos = 0

    for m in TAG_RE.finditer(text):
        out.append(text[pos:m.start()])
        out.append(" ")
        pos = m.end()

        tag = m.group(1).strip()
        key = ALIASES.get(tag.lower(), tag.lower())
        if key in MOODS:
            mood = key
        elif key in DELIVERY:
            delivery = key
        elif key not in _warned_tags:
            _warned_tags.add(key)
            _LOGGER.warning("dropping %s tag %r (logged once)",
                            "unsupported" if key in UNSUPPORTED else "unknown", tag)

    out.append(text[pos:])
    clean = " ".join("".join(out).split())
    # A tag between a word and its punctuation leaves "word ." behind
    clean = re.sub(r"\s+([,.!?;:])", r"\1", clean)
    return clean, mood, delivery


def has_mood_tag(text: str) -> bool:
    return any(ALIASES.get(k, k) in MOODS
               for k in (m.group(1).strip().lower() for m in TAG_RE.finditer(normalize(text))))


def mood_vector(mood: Optional[str], strength: float) -> Optional[list[float]]:
    """8-dim emotion vector for a mood; None = the reference clip's own emotion."""
    if mood is None:
        return None
    weights = MOODS[mood]
    return [round(weights.get(d, 0.0) * strength, 4) for d in EMO_DIMS]


def estimate_seconds(text: str) -> float:
    """Very rough spoken-duration estimate for merging decisions."""
    return len(text.split()) / WORDS_PER_SECOND


def norm_text(text: str) -> str:
    """Tag-free, whitespace- and case-insensitive form for duplicate checks."""
    return " ".join(TAG_RE.sub(" ", text).split()).lower()


# ---------------------------------------------------------------------------
# Model wrapper — the ONLY place that talks to IndexTTS
# ---------------------------------------------------------------------------

class SynthEngine:
    """Loads IndexTTS-2.5 once and exposes a blocking synthesize_iter()."""

    def __init__(
        self,
        model_dir: str,
        prompt_wav: str,
        lang: str,
        bf16: bool,
        num_beams: int,
        duration_factor: float,
        emo_strength: float,
        default_mood: Optional[str],
    ) -> None:
        self.model_dir = model_dir
        self.prompt_wav = prompt_wav
        self.lang = lang
        self.bf16 = bf16
        self.num_beams = num_beams
        self.duration_factor = duration_factor
        self.emo_strength = emo_strength
        self.default_mood = default_mood

        self.tts: Any = None
        self.sample_rate = SAMPLE_RATE
        self._lock = threading.Lock()
        self._fade: Any = None

    def load(self) -> None:
        if not os.path.exists(self.prompt_wav):
            raise RuntimeError(f"prompt WAV not found: {self.prompt_wav!r}")

        from indextts.infer_v2_5 import IndexTTS2  # noqa: PLC0415
        from indextts.utils.common import fade_out_tail  # noqa: PLC0415

        self._fade = fade_out_tail
        _LOGGER.info("loading IndexTTS-2.5 from %s (bf16=%s)", self.model_dir, self.bf16)
        t0 = time.monotonic()
        # use_cuda_kernel=False: the BigVGAN CUDA kernel needs ninja to build
        # and BigVGAN is only ~0.13 s of a ~3 s sentence anyway.
        self.tts = IndexTTS2(
            cfg_path=os.path.join(self.model_dir, "config.yaml"),
            model_dir=self.model_dir,
            use_bf16=self.bf16,
            use_cuda_kernel=False,
        )
        _LOGGER.info("model ready in %.1fs (VRAM %.2f GB allocated)",
                     time.monotonic() - t0, torch.cuda.memory_allocated() / 2**30)
        self._warmup()

    def _warmup(self) -> None:
        """Default mood and one other mood: the cold first call fills the
        reference-voice caches and CUDA setup (~6 s vs ~2.5 s warm)."""
        moods = [self.default_mood, "happy" if self.default_mood != "happy" else "calm"]
        for mood in moods:
            t0 = time.monotonic()
            try:
                for _ in self.synthesize_iter(WARMUP_TEXT, mood, None):
                    pass
            except Exception:
                _LOGGER.warning("warmup synthesis failed", exc_info=True)
                return
            _LOGGER.info("warmup (%s) done in %.1fs", mood, time.monotonic() - t0)

    def synthesize_iter(
        self, text: str, mood: Optional[str], delivery: Optional[str]
    ) -> Iterator[tuple[bytes, int]]:
        """Blocking generator: yields (pcm16 chunk, sample_rate) per text
        segment as the model finishes it. Run it in a worker thread and
        forward each chunk immediately."""
        vec = mood_vector(mood or self.default_mood, self.emo_strength)
        dur = self.duration_factor * DELIVERY.get(delivery or "", 1.0)
        dur = max(0.5, min(2.0, dur))
        _LOGGER.info("model call: mood=%s emo_vector=%s duration_factor=%.2f",
                     mood or self.default_mood, vec, dur)

        t0 = time.monotonic()
        samples = 0
        with self._lock, torch.no_grad():
            gen = self.tts.infer(
                self.prompt_wav, text, None, self.lang,
                emo_vector=vec, emo_alpha=1.0,
                stream_return=True, duration_factor=dur,
                num_beams=self.num_beams,
            )
            # The generator yields speech, then a 200 ms silence, per
            # segment. Hold one item back so the final silence is dropped.
            held: Optional[torch.Tensor] = None
            for item in gen:
                if not torch.is_tensor(item):
                    continue
                if held is not None:
                    samples += held.shape[-1]
                    yield self._pcm(held), self.sample_rate
                held = item
            if held is not None and held.abs().max() > 0:
                samples += held.shape[-1]
                yield self._pcm(held), self.sample_rate

        gen_s = time.monotonic() - t0
        audio_s = samples / self.sample_rate
        _LOGGER.info("block done: audio %.2fs, generation %.2fs, RTF %.2f",
                     audio_s, gen_s, gen_s / audio_s if audio_s else float("nan"))

    def _pcm(self, wav: torch.Tensor) -> bytes:
        """IndexTTS PCM-scaled float [1, N] -> little-endian PCM16 bytes.
        Fades the last 20 ms: the non-stream path does this to hide a click
        when the stop token is sampled early; the stream path doesn't."""
        x = self._fade(wav.float(), self.sample_rate)
        return x.clamp(-32767.0, 32767.0).squeeze(0).to(torch.int16).numpy().tobytes()


# ---------------------------------------------------------------------------
# Wyoming handler
# ---------------------------------------------------------------------------

class BridgeHandler(AsyncEventHandler):
    def __init__(
        self,
        *args: Any,
        engine: SynthEngine,
        info: Info,
        ready: asyncio.Event,
        min_merged_seconds: float,
        prebuffer_seconds: float,
        samples_per_chunk: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.engine = engine
        self.info = info
        self.sample_rate = engine.sample_rate
        self._ready = ready
        self._min_merged = min_merged_seconds
        self._prebuffer = prebuffer_seconds
        self._samples_per_chunk = samples_per_chunk

        self._stash = ""
        self._buf = ""
        self._got_chunks = False
        self._audio_started = False
        self._mood: Optional[str] = None
        self._delivery: Optional[str] = None

        # Sentence merging accumulator
        self._pending_sentences: list[str] = []
        self._first_queued = False

        # Duplicate detection for the one-shot synthesize HA sends with a stream
        self._streaming = False
        self._stream_text = ""
        self._last_stream_norm = ""
        self._last_stream_end = 0.0

        self._work_q: Optional[asyncio.Queue] = None
        self._play_q: Optional[asyncio.Queue] = None
        self._fetch_task: Optional[asyncio.Task] = None
        self._play_task: Optional[asyncio.Task] = None

    # -- sentence splitter -------------------------------------------------

    def _take_sentences(self, text: str) -> list[str]:
        self._buf += text
        out: list[str] = []
        pos = 0
        while (m := _TERM_RE.search(self._buf, pos)) is not None:
            word = _WORD_RE.search(self._buf[: m.start()])
            if word and word.group(0).lower() in _ABBREVIATIONS:
                pos = m.start() + 1
                continue
            sentence = self._buf[: m.end()].strip()
            self._buf = self._buf[m.end():]
            pos = 0
            if sentence:
                out.append(sentence)
        return out

    # -- sentence merging ----------------------------------------------------

    def _feed_sentence(self, sentence: str) -> None:
        """Accumulate sentences; queue a merged block once the estimated
        speech length exceeds the merge threshold. Hard cap of 6 sentences
        bounds onset latency for fragment-heavy replies. A sentence with a
        mood tag starts a new block, so the new mood doesn't colour the
        sentences before it."""
        if self._pending_sentences and has_mood_tag(sentence):
            self._flush_pending()
        self._pending_sentences.append(sentence)
        if not self._first_queued:
            # LATENCY-FIRST: don't wait to merge the opening sentence
            self._first_queued = True
            self._flush_pending()
            return
        est = sum(estimate_seconds(s) for s in self._pending_sentences)
        if est >= self._min_merged or len(self._pending_sentences) >= 6:
            self._flush_pending()

    def _flush_pending(self) -> None:
        if self._pending_sentences:
            merged = " ".join(self._pending_sentences)
            self._pending_sentences = []
            self._work_q.put_nowait(merged)

    # -- pipeline lifecycle --------------------------------------------------

    async def _start_pipeline(self) -> None:
        self._work_q = asyncio.Queue()
        self._play_q = asyncio.Queue()
        self._fetch_task = asyncio.create_task(self._fetch_worker())
        self._play_task = asyncio.create_task(self._play_worker())

    async def _stop_pipeline(self) -> None:
        if self._fetch_task is None:
            return
        self._flush_pending()
        await self._work_q.put(None)
        await self._fetch_task
        await self._play_q.put(None)
        await self._play_task
        self._fetch_task = None
        self._play_task = None
        self._work_q = None
        self._play_q = None

    # -- workers ---------------------------------------------------------------

    async def _fetch_worker(self) -> None:
        """Synthesize merged blocks off the event loop, one at a time.
        A failed block is reported as a Wyoming Error; the rest still run."""
        await self._ready.wait()

        while (text := await self._work_q.get()) is not None:
            clean, mood, delivery = parse_tags(text)
            if mood:
                self._mood = mood
            if delivery:
                self._delivery = delivery
            if not re.search(r"\w", clean):
                continue

            _LOGGER.info("synthesizing: text=%r", clean[:60])

            try:
                await self._stream_block(clean, self._mood, self._delivery)
            except Exception as err:
                _LOGGER.exception("block synthesis failed: %r", clean[:60])
                try:
                    await self.write_event(
                        Error(
                            text=f"synthesis failed: {err}",
                            code=err.__class__.__name__,
                        ).event()
                    )
                except Exception:
                    pass  # connection may already be gone
                continue

    async def _stream_block(
        self, text: str, mood: Optional[str], delivery: Optional[str]
    ) -> None:
        """Run the model in a worker thread and hand every chunk to the play
        queue the moment it is produced."""
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        def produce() -> None:
            try:
                for item in self.engine.synthesize_iter(text, mood, delivery):
                    loop.call_soon_threadsafe(q.put_nowait, item)
            except BaseException as err:  # noqa: BLE001
                loop.call_soon_threadsafe(q.put_nowait, err)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        fut = loop.run_in_executor(None, produce)
        error: Optional[BaseException] = None
        first = True
        t0 = time.monotonic()
        while (item := await q.get()) is not None:
            if isinstance(item, BaseException):
                error = item
                continue
            audio, rate = item
            if audio:
                if first:
                    _LOGGER.info("first audio chunk after %.2fs",
                                 time.monotonic() - t0)
                    first = False
                await self._play_q.put((audio, rate))
        await fut
        if error is not None:
            raise error

    async def _play_worker(self) -> None:
        """Write synthesized audio to the Wyoming client: hold AudioStart
        until --prebuffer-seconds is queued, then re-emit in fixed-size
        chunks, flushing the remainder once at the end."""
        width = 2
        bytes_per_chunk = width * self._samples_per_chunk

        buf = bytearray()
        rate: Optional[int] = None
        started = False
        buffered_bytes = 0

        async def _maybe_emit_start() -> None:
            nonlocal started
            if not self._audio_started and buf:
                await self.write_event(
                    AudioStart(rate=rate, width=width, channels=1).event()
                )
                self._audio_started = True
            started = True

        while (item := await self._play_q.get()) is not None:
            audio, r = item
            if not audio:
                continue
            if rate is None:
                rate = r

            buf.extend(audio)
            buffered_bytes += len(audio)

            if not started:
                if buffered_bytes < int(self._prebuffer * 2 * rate):
                    continue
                await _maybe_emit_start()

            while len(buf) >= bytes_per_chunk:
                await self.write_event(
                    AudioChunk(
                        audio=bytes(buf[:bytes_per_chunk]),
                        rate=rate,
                        width=width,
                        channels=1,
                    ).event()
                )
                del buf[:bytes_per_chunk]

        # Short utterances that never reached the prebuffer are emitted here.
        if buf:
            await _maybe_emit_start()
            await self.write_event(
                AudioChunk(
                    audio=bytes(buf), rate=rate, width=width, channels=1
                ).event()
            )
            buf.clear()

    # -- wyoming event handling ---------------------------------------------

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.info.event())
            return True

        if SynthesizeStart.is_type(event.type):
            self._reset_utterance()
            self._streaming = True
            self._stream_text = ""
            await self._start_pipeline()
            return True

        if SynthesizeChunk.is_type(event.type):
            self._got_chunks = True
            chunk_text = SynthesizeChunk.from_event(event).text
            self._stream_text += chunk_text
            for sentence in self._take_sentences(chunk_text):
                self._feed_sentence(sentence)
            return True

        if SynthesizeStop.is_type(event.type):
            if self._got_chunks:
                tail = self._buf.strip()
                self._buf = ""
                if tail:
                    self._feed_sentence(tail)
            else:
                text = self._stash
                self._stash = ""
                self._stream_text = text
                if text.strip():
                    self._feed_sentence(text.strip())

            await self._finish_utterance()

            self._streaming = False
            self._last_stream_norm = norm_text(self._stream_text)
            self._last_stream_end = time.monotonic()
            await self.write_event(SynthesizeStopped().event())
            return True

        if Synthesize.is_type(event.type):
            text = Synthesize.from_event(event).text or ""

            # (1) Compat one-shot that HA sends alongside a stream: ignore
            #     (but keep it as the text for a start/stop with no chunks).
            if self._streaming:
                self._stash = text
                return True

            # (2) Late duplicate right after a finished stream.
            if (self._last_stream_norm
                    and time.monotonic() - self._last_stream_end
                    < LATE_DUPLICATE_WINDOW_S
                    and norm_text(text) == self._last_stream_norm):
                _LOGGER.warning(
                    "late duplicate of finished stream suppressed: %r",
                    text[:60])
                return True

            # (3) Genuine non-streaming request (tts.speak, "Try voice", ...)
            self._reset_utterance()
            await self._start_pipeline()
            for sentence in self._take_sentences(text + " "):
                self._feed_sentence(sentence)
            tail = self._buf.strip()
            self._buf = ""
            if tail:
                self._feed_sentence(tail)
            await self._finish_utterance(always_audio=True)
            return True

        return True

    def _reset_utterance(self) -> None:
        self._stash = ""
        self._buf = ""
        self._got_chunks = False
        self._audio_started = False
        self._mood = None
        self._delivery = None
        self._pending_sentences = []
        self._first_queued = False

    async def _finish_utterance(self, always_audio: bool = False) -> None:
        """Drain the pipeline and close the audio stream. A one-shot client
        waits for audio-stop, so always_audio sends an empty start/stop pair
        when nothing was synthesized (blank text, every block failed)."""
        await self._stop_pipeline()
        if not self._audio_started and always_audio:
            await self.write_event(
                AudioStart(rate=self.sample_rate, width=2, channels=1).event()
            )
            self._audio_started = True
        if self._audio_started:
            await self.write_event(AudioStop().event())
            self._audio_started = False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IndexTTS-2.5 Wyoming bridge")
    p.add_argument("--uri", default="tcp://0.0.0.0:10300", help="Wyoming server URI")
    p.add_argument("--repo", default="repo",
                   help="Path to the index-tts checkout (the bridge chdirs there: "
                   "the model resolves ./checkpoints relative to it)")
    p.add_argument("--model-dir", default=None,
                   help="Checkpoint dir; defaults to <repo>/checkpoints")
    p.add_argument("--prompt-wav", required=True, help="Reference speaker WAV path")
    p.add_argument("--lang", default="EN", help="Model language code (EN, ZH, JA, ES, AR)")
    p.add_argument("--default-mood", default="warm",
                   help="Mood for text before any mood tag ('' or 'none' = the "
                   f"reference clip's own emotion). One of: {', '.join(MOODS)}")
    p.add_argument("--emo-strength", type=float, default=1.0,
                   help="Multiplier on every mood vector in MOODS")
    p.add_argument("--speed", type=float, default=1.0,
                   help="1.0 = normal (sent as duration_factor = 1/speed)")
    p.add_argument("--num-beams", type=int, default=1,
                   help="GPT beam search width (model default 3; 1 is ~0.2 s "
                   "faster per sentence and was used for the approved samples)")
    p.add_argument("--no-bf16", action="store_true", help="Run in fp32")
    p.add_argument("--min-merged-seconds", type=float, default=2.5,
                   help="Batch sentences after the first until the estimated "
                   "speech length reaches this many seconds (0 disables merging)")
    p.add_argument("--prebuffer-seconds", type=float, default=1.0,
                   help="Hold playback until this much audio is queued (0 disables)")
    p.add_argument("--samples-per-chunk", type=int, default=2048,
                   help="Re-emit AudioChunks of this many samples (~93 ms at 22.05 kHz)")
    p.add_argument("--voice", default="nicoleheart", help="Voice name reported to HA")
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    repo = Path(args.repo).resolve()
    if not (repo / "indextts").is_dir():
        raise SystemExit(f"index-tts repo not found: {repo}")
    model_dir = str(Path(args.model_dir).resolve()) if args.model_dir else str(repo / "checkpoints")
    prompt_wav = str(Path(args.prompt_wav).resolve())
    # infer_v2_5 sets HF_HUB_CACHE=./checkpoints/hf_cache at import time
    os.chdir(repo)
    sys.path.insert(0, str(repo))

    mood = args.default_mood.strip().strip("[]").lower()
    mood = ALIASES.get(mood, mood)
    if mood in ("", "none"):
        default_mood = None
    elif mood in MOODS:
        default_mood = mood
    else:
        raise SystemExit(f"unknown --default-mood {args.default_mood!r}; "
                         f"choose from: {', '.join(MOODS)}")
    if args.speed <= 0:
        raise SystemExit("--speed must be > 0")
    _LOGGER.info("mood for untagged text: %s", default_mood or "reference clip")

    engine = SynthEngine(
        model_dir=model_dir,
        prompt_wav=prompt_wav,
        lang=args.lang,
        bf16=not args.no_bf16,
        num_beams=args.num_beams,
        duration_factor=1.0 / args.speed,
        emo_strength=args.emo_strength,
        default_mood=default_mood,
    )

    ready = asyncio.Event()

    async def _load_engine() -> None:
        try:
            await asyncio.to_thread(engine.load)
        except Exception:
            _LOGGER.critical("model load failed", exc_info=True)
            os._exit(1)
        _LOGGER.info("engine ready — synthesis enabled")
        ready.set()

    loader = asyncio.create_task(_load_engine())  # noqa: F841

    attribution = Attribution(
        name="bilibili IndexTeam",
        url="https://github.com/index-tts/index-tts",
    )
    info = Info(
        tts=[
            TtsProgram(
                name="indextts",
                attribution=attribution,
                installed=True,
                description="IndexTTS-2.5 (in-process)",
                version="2.5",
                voices=[
                    TtsVoice(
                        name=args.voice,
                        attribution=attribution,
                        installed=True,
                        description=f"IndexTTS-2.5 cloned voice ({args.voice})",
                        version="2.5",
                        languages=[args.lang.lower()],
                    )
                ],
                supports_synthesize_streaming=True,
            )
        ]
    )

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("listening on %s (model=%s) — loading model in background",
                 args.uri, model_dir)

    server_task = asyncio.create_task(
        server.run(
            partial(
                BridgeHandler,
                engine=engine,
                info=info,
                ready=ready,
                min_merged_seconds=args.min_merged_seconds,
                prebuffer_seconds=args.prebuffer_seconds,
                samples_per_chunk=args.samples_per_chunk,
            )
        )
    )
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, server_task.cancel)
    try:
        await server_task
    except asyncio.CancelledError:
        _LOGGER.info("server stopped")


if __name__ == "__main__":
    asyncio.run(main())
