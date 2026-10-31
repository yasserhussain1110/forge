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

- **Hardware:** Weeks 1–4 run on a laptop or Google Colab. From week 5 you need one rented GPU, such as an A10G or L4 on RunPod, Lambda or Modal. Week 21 needs two GPUs.
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
