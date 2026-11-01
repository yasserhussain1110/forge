# Forge: 6-Month Plan

Goal: build an LLM inference engine from scratch, learning AI and GPU fundamentals along the way, and use it to land an AI infrastructure role.

## How to use this folder

Each morning, open the next day file, for example `week-01/day-001.md`, and do only what it says. Tick the checkboxes inside it as you go. When you finish the day, tick it in [PROGRESS.md](PROGRESS.md). When a day spills over, finish it the next morning and keep going; the dates don't matter, the order does.

- Each `week-NN/` folder holds that week's seven day files.
- Days 1–6 of each week are working days (about 2–3 hours each).
- Day 7 of each week (day-007, day-014, ...) is a Sunday review: catch up, reflect, and write a short weekly log.

## Phases

| Days | Weeks | Theme |
|---|---|---|
| 001–028 | 1–4 | Foundations: PyTorch, neural networks, transformers, Llama architecture |
| 029–063 | 5–9 | Forge v0: your own Llama engine, KV cache, server, benchmarks |
| 064–091 | 10–13 | GPUs and kernels: GPU architecture, Triton, FlashAttention |
| 092–126 | 14–18 | Serving many users: continuous batching, PagedAttention, prefix caching |
| 127–154 | 19–22 | Advanced: speculative decoding, quantization, multi-GPU |
| 155–182 | 23–26 | Launch: benchmarks, open source, interview prep, applications |

## Ground rules

- **Hardware:** Everything runs on your own GPU (an RTX 4060 with 8 GB) unless a day file explicitly says to rent. The plan uses Llama 3.2 1B and 3B, which fit in 8 GB. Avoid renting:
  - **Required rental:** one session on a machine with two GPUs in week 21 (day 145), because tensor parallelism can't be measured on one GPU.
  - **Optional rentals:** day 131 (real speculative-decoding speed-up with Llama 3.1 8B) and day 156 (rerunning the final benchmark on a datacenter GPU). Skip them if cost matters; each day says what to do instead.
  - **When you do rent:** build and test everything locally first, have your scripts ready, rent by the hour from the cheapest provider that has the GPU (RunPod, Vast.ai, Lambda or Modal), and shut the machine down as soon as your numbers are recorded.
- **Coding interviews:** From week 14, add 30 minutes a day of practice using the [NeetCode 150](https://neetcode.io/practice) list.
- **Write as you go:** keep a `notes/` folder and a `RESULTS.md` file in the repo from day one. Every number you measure goes into `RESULTS.md`.
- **If you fall behind:** Take the time from week 22, which is optional, instead of skipping writing. The design docs and blog posts are what hiring managers will actually read.
- **Expectations:** Starting from zero AI experience, this plan makes you a credible inference engineer in 6 months. Whether you're hired at staff level depends mostly on your existing career history. The project and the writing will at least get you serious interviews.

## Deliverables checklist

- [ ] Blog post 1: Building a Llama inference engine from scratch, part 1 (day 041)
- [ ] Design doc 1: Forge v0 architecture and its limits (day 055)
- [ ] Blog post 2: Where decode time goes (day 062)
- [ ] Blog post 3: Writing attention kernels in Triton as a beginner (day 090)
- [ ] Design doc 2: Scheduler and request lifecycle (day 095)
- [ ] Blog post 4: Implementing PagedAttention from scratch (day 111)
- [ ] Design doc 3: Scheduler retrospective (day 125)
- [ ] Blog post 5: Running one model across several GPUs (day 146)
- [ ] Design doc 4: Separating prefill and decode (day 150)
- [ ] Final blog post, demo video and public launch (days 155–167)
- [ ] At least one pull request to vLLM, SGLang, PyTorch or Triton (days 163–166)
