"""Exercise a running Wyoming TTS bridge the way Home Assistant does.

Covers: a cold first request, a plain one-shot `synthesize`, a stream with the
compat `synthesize` HA sends during it, a late duplicate right after a stream,
and a blank one-shot. Needs only the `wyoming` package:

    python test_bridge.py --host 127.0.0.1 --port 10214 \
        --tagged "[happy] Hello! [laughter] This is a test."

Expected: audio for every request except the late duplicate (TIMEOUT, i.e.
suppressed) and the blank one-shot (an empty audio-start/audio-stop pair).
A stream that ends without `synthesize-stopped`, a one-shot without
`audio-stop`, or audio for the late duplicate are bugs.
"""
import argparse
import asyncio
import time

from wyoming.audio import AudioChunk
from wyoming.client import AsyncTcpClient
from wyoming.tts import Synthesize, SynthesizeChunk, SynthesizeStart, SynthesizeStop


async def collect(c, until, timeout):
    """Read events until type `until`. Returns (first-audio s, audio s, types)."""
    t = time.monotonic()
    first, seconds, types = None, 0.0, []
    try:
        while True:
            ev = await asyncio.wait_for(c.read_event(), timeout)
            if ev is None:
                break
            types.append(ev.type)
            if AudioChunk.is_type(ev.type):
                ch = AudioChunk.from_event(ev)
                seconds += len(ch.audio) / (ch.width * ch.channels * ch.rate)
                first = first if first is not None else time.monotonic() - t
            if ev.type == until:
                break
    except asyncio.TimeoutError:
        types.append("TIMEOUT")
    return first, seconds, types


def summary(types):
    out = []
    for t in types:
        if out and out[-1][0] == t:
            out[-1][1] += 1
        else:
            out.append([t, 1])
    return " ".join(f"{t}x{n}" if n > 1 else t for t, n in out)


def line(label, first, seconds, types):
    f = f"first {first:5.2f}s" if first is not None else "first   -   "
    print(f"{label:22s} {f}  audio {seconds:4.1f}s | {summary(types)}")


async def oneshot(args, text):
    async with AsyncTcpClient(args.host, args.port) as c:
        await c.write_event(Synthesize(text=text).event())
        return await collect(c, "audio-stop", args.timeout)


async def stream(args, text, late_dup=False):
    """Mimic HA: start, chunks, compat one-shot, stop (+ optional late dup)."""
    async with AsyncTcpClient(args.host, args.port) as c:
        await c.write_event(SynthesizeStart().event())
        half = len(text) // 2
        for part in (text[:half], text[half:]):  # deltas carry their own spaces
            await c.write_event(SynthesizeChunk(text=part).event())
        await c.write_event(Synthesize(text=text).event())
        await c.write_event(SynthesizeStop().event())
        res = await collect(c, "synthesize-stopped", args.timeout)
        dup = None
        if late_dup:
            await c.write_event(Synthesize(text=text).event())
            dup = await collect(c, "audio-stop", 3.0)
        return res, dup


async def main(args):
    line("cold one-shot", *await oneshot(args, "First request after restart."))
    line("one-shot (tagged)", *await oneshot(args, args.tagged))
    res, _ = await stream(args, args.stream_text)
    line("stream + compat", *res)
    res, dup = await stream(args, "The garage door is open.", late_dup=True)
    line("stream", *res)
    line("  late duplicate", *dup)
    line("blank one-shot", *await oneshot(args, "   "))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=10200)
    ap.add_argument("--tagged", default="Hello! This is a one-shot request.",
                    help="one-shot text; put the model's own tags in it")
    ap.add_argument("--stream-text", default="The lights are off. Good night!")
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="per-event timeout; covers a model still loading")
    asyncio.run(main(ap.parse_args()))
