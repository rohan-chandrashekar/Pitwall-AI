# PitWall-AI

F1 race strategy prediction engine using a 4-agent LangGraph pipeline.

## Quick Reference

- Run replay: `python -m pitwall --race bahrain-2024 --driver VER --speed 1000`
- Run tests: `python -m pytest tests/ -x -q`
- Train model: `python -m pitwall --train --seasons 2023 2024 2025`

## Architecture

Sequential per-lap pipeline: Scout → Spy → Strategist → Principal
Ghost Car Evaluator runs alongside to validate strategy quality.

## Key Conventions

- Package is `pitwall/` (run with `python -m pitwall`)
- All agents are in `pitwall/agents/`
- Models (PyTorch tire model, overtake model) in `pitwall/models/`
- OpenF1 API client in `pitwall/data/openf1_client.py`
- Tests use NO network calls — mock the OpenF1 client
- Unit tests must run in <10 seconds total
- InfluxDB logging at DEBUG level (optional infrastructure)
- Never cap ghost car deltas — fix root cause instead
- GA must enforce 2+ compound diversity at EVERY level (creation, repair, crossover, mutation)
- First stint compound is always locked to current compound
- Clamp math.exp exponents to [-500, 500] to prevent overflow
- remaining_laps = total_laps - current_lap + 1
