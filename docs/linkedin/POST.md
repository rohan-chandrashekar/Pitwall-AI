# LinkedIn Post — PitWall-AI

## 📝 Post text (copy-paste ready)

---

🏎️ What if an **Agentic AI** system could make Formula 1 race-strategy decisions lap-by-lap — the way a real pitwall does?

I spent the last few weeks building exactly that. Meet **PitWall-AI** ⚡

🤖 **Five specialised AI agents**, orchestrated through LangGraph, cooperating every single lap of a Grand Prix:

🚦 **Scout** — ingests live telemetry from the OpenF1 API
🕵️ **Spy** — predicts when rivals will pit (Bayesian hazard model per compound)
🧠 **Strategist** — evolves optimal tire strategies via a DEAP genetic algorithm on top of a PyTorch tire-degradation network
👻 **Ghost Car** — scores the AI's plan against what actually happened, using a relative-delta design that's immune to model drift
🎙️ **Principal** — writes race-engineer radio briefings & post-race debriefs via LLaMA-3.3-70B (Groq)

📈 **What it pulled off**

• Simulates any 2023-2025 Grand Prix at up to **1000× replay speed**, or runs on **live** sessions
• Validates compound differentiation ≥ **0.5 s/lap** (HARD vs SOFT) before shipping a model — refuses to deploy otherwise
• Genetic algorithm enforces **FIA 2-compound diversity at every operator** (creation, crossover, mutation, repair)
• Trained on **3 full seasons** of data via OpenF1 + FastF1 (with rainfall, humidity & track-wetness features)
• **32-panel Grafana dashboard** backed by InfluxDB — live AI-vs-actual lap deltas, stint-by-stint strategy comparison, streaming race-engineer radio
• **5,384 LOC** Python, **719 LOC** of deterministic tests running in < 10 s (no network, zero flakes)

💡 **What I learned building an agentic system**

→ Clean state boundaries between agents matter more than the agents themselves
→ Every AI layer needs a physics-based fallback for when the LLM / model fails
→ Relative deltas beat absolute predictions — errors cancel, pure signal remains
→ If your ghost delta looks wrong, **never cap it** — fix the root cause

The visuals below show the dashboard running a HAM Melbourne simulation — the AI plan matched the actual compound choice but would have pitted one lap earlier on each stop.

🔗 Full write-up + code: github.com/<your-username>/pitwall-ai

#AgenticAI #Formula1 #F1 #LangGraph #MachineLearning #PyTorch #LLM #Python #AIEngineering #Grafana #GeneticAlgorithms #DataScience

---

## 📸 Images to attach (in this order)

1. **01_full_dashboard.png** — Hero shot. Shows the full 32-panel dashboard running HAM Melbourne live. This is the single most impressive visual.
2. **02_strategy_comparison.png** — The stint-by-stint AI Plan vs Actual table with colour-coded compounds + lap ranges. Tells the story at a glance.
3. **09_lap_times.png** — AI-predicted vs actual lap times (y-axis in mm:ss). Shows the AI tracking real performance lap-by-lap.
4. **08_header_row.png** — The top of the dashboard showing the flag, circuit, driver, date, total laps. Eye-catching and contextual.

LinkedIn allows up to 20 images per post; 4 is the engagement sweet spot. Post them in the order above — the full dashboard should lead.

## 📏 Character counts

- Post text: ~2,050 characters (LinkedIn limit 3,000)
- Well under the "show more" fold (210 chars) for the hook
- Optimised for skim-readability with emoji bullets
