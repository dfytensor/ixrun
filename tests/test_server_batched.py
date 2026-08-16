"""E2E: batched server — 8 concurrent streaming requests, measure wall time."""
import json, time, threading, urllib.request

BASE = "http://127.0.0.1:8100"
PROMPTS = [
    "Write a 60-word story about a lighthouse keeper.",
    "Explain photosynthesis in simple terms.",
    "List three interesting facts about Mars.",
    "Describe the water cycle briefly.",
    "What causes the seasons on Earth?",
    "Summarize the plot of Romeo and Juliet.",
    "How does a refrigerator work?",
    "Give a short history of the internet.",
]

def one(prompt, out, idx):
    body = json.dumps({
        "model": "minicpm5",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 80, "stream": True,
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    n = 0
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d["choices"][0]["delta"].get("content"):
                n += 1
    out[idx] = (time.time() - t0, n)

out = {}
threads = [threading.Thread(target=one, args=(p, out, i))
           for i, p in enumerate(PROMPTS)]
t0 = time.time()
for t in threads: t.start()
for t in threads: t.join()
wall = time.time() - t0
total_chunks = sum(n for _, n in out.values())
print(f"8 concurrent streaming reqs: wall={wall:.1f}s, chunks={total_chunks}")
print(f"aggregate throughput ~= {8*80/wall:.1f} tok/s (80 max_new each)")
for i in sorted(out):
    dt, n = out[i]
    print(f"  req{i}: {dt:.1f}s {n} chunks")
