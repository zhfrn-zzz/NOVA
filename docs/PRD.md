# NOVA — Product Requirements Document (PRD)

**Version:** 1.0
**Author:** Zhafran
**Date:** February 2026
**Status:** Draft

---

## 1. Executive Summary

NOVA (Neural-Orchestrated Voice Assistant) is a personal AI voice assistant designed to run on resource-constrained hardware (ASUS E410MA, Celeron N4020, 4GB RAM) with Ubuntu Linux. The system leverages cloud-based AI APIs for all computationally intensive operations — speech recognition, natural language processing, and speech synthesis — while the local machine serves exclusively as a lightweight orchestrator.

The project addresses a core challenge: building a functional, voice-driven AI assistant comparable to commercial solutions (Google Assistant, Alexa) without requiring expensive hardware or paid subscriptions, using only free-tier cloud API services.

---

## 2. Problem Statement

### 2.1 Pain Points

Commercial voice assistants are locked into proprietary ecosystems, offer limited customization, and require specific hardware or premium subscriptions for advanced AI features. Meanwhile, running local LLMs on budget hardware is computationally infeasible. Users with low-spec machines are excluded from the AI assistant experience entirely.

### 2.2 Target User

A technically proficient individual (student/developer) with a low-spec Ubuntu laptop who wants a customizable, voice-controlled AI assistant for daily productivity, learning, and system automation — without financial investment beyond internet connectivity.

---

## 3. Product Vision & Goals

### 3.1 Vision

A fully voice-interactive, open-source personal AI assistant that runs on any Linux machine with an internet connection, powered entirely by free-tier cloud APIs with intelligent failover.

### 3.2 Goals

| Goal | Success Metric | Priority |
|------|---------------|----------|
| Functional voice pipeline | End-to-end voice interaction works reliably | P0 (Must Have) |
| Response latency < 5 seconds | Average STT + LLM + TTS < 5s on stable connection | P0 (Must Have) |
| Zero cost operation | No paid APIs or subscriptions required | P0 (Must Have) |
| Multi-provider failover | Auto-switch when primary API hits rate limit | P1 (Should Have) |
| System control integration | Volume, brightness, app launch via voice | P1 (Should Have) |
| Conversation memory | Context maintained within session (sliding window) | P1 (Should Have) |
| Wake word activation | Hands-free activation via hotword | P2 (Nice to Have) |
| Web search capability | Answer current-events questions | P2 (Nice to Have) |
| Smart home integration | Home Assistant API connection | P3 (Future) |

---

## 4. Hardware & Environment Constraints

### 4.1 Target Hardware

| Component | Specification | Implication |
|-----------|--------------|-------------|
| CPU | Intel Celeron N4020 (2C/2T, 1.1–2.8GHz) | No local ML inference; all AI must be cloud-based |
| RAM | 4GB DDR4 (soldered, non-upgradable) | Orchestrator + OS must fit within ~3GB (Ubuntu idle ~900MB) |
| Storage | 128GB eMMC | Sufficient for code; no model storage needed |
| Audio | Built-in mic + speakers | Adequate for basic voice I/O; external mic recommended |
| Network | Wi-Fi 802.11ac | Critical dependency — all AI services require internet |
| OS | Ubuntu Linux (latest LTS) | Low overhead; full Python ecosystem available |

### 4.2 Resource Budget

| Resource | Total | OS Overhead | Available for NOVA | Target |
|----------|-------|-------------|-------------------|--------|
| RAM | 4096 MB | ~900 MB | ~3100 MB | < 300 MB |
| CPU (idle) | 100% | ~5–10% | ~90% | < 15% sustained |
| Storage | 128 GB | ~12 GB | ~116 GB | < 500 MB |
| Network | Variable | N/A | All | < 1 MB per interaction |

---

## 5. System Architecture

### 5.1 Overview

NOVA follows a **thin-client architecture**. The local machine functions as an orchestrator managing audio I/O, API routing, state management, and user interaction. All computationally intensive AI operations are offloaded to free-tier cloud APIs.

### 5.2 Core Pipeline

Estimated total latency: 2–5 seconds under optimal network conditions.

| Stage | Component | Provider | Est. Latency | Fallback |
|-------|-----------|----------|-------------|----------|
| 1. Wake Word | Hotword Detection | Porcupine (local) | ~instant | Keyboard trigger |
| 2. Audio Capture | Microphone Recording | PyAudio (local) | 1–3s (user speech) | N/A |
| 3. STT | Speech-to-Text | Groq Whisper API | 0.3–1s | Google STT Free |
| 4. LLM | Intent + Response | Gemini 2.5 Flash | 1–3s | Groq Llama 3.3 70B |
| 5. TTS | Text-to-Speech | Edge TTS | 0.3–0.5s | Cloudflare Workers AI |
| 6. Playback | Speaker Output | mpv (local) | Realtime | aplay |

### 5.3 Provider Details

#### 5.3.1 STT Providers

| Provider | Model | Free Tier Limit | Latency | Priority |
|----------|-------|----------------|---------|----------|
| Groq | whisper-large-v3-turbo | Limited RPM (very fast) | ~0.3s | Primary |
| Google Cloud STT | Default | 60 min/month | ~1–2s | Fallback |
| Cloudflare Workers AI | Whisper | Included in free tier | ~1–2s | Fallback 2 |

#### 5.3.2 LLM Providers

| Provider | Model | Free Tier Limit | Strengths | Priority |
|----------|-------|----------------|-----------|----------|
| Google Gemini | 2.5 Flash / 2.0 Flash | 15 RPM, 1500 req/day | Fast, function calling, generous | Primary |
| Groq | Llama 3.3 70B | 30 RPM, 6000 TPD | Ultra-fast inference | Fallback 1 |
| Cloudflare Workers AI | Llama/Mistral | 10K neurons/day | No per-minute limit | Fallback 2 |
| OpenRouter | Various free models | Varies | Model diversity | Fallback 3 |

#### 5.3.3 TTS Providers

| Provider | Method | Free Tier | Quality | Priority |
|----------|--------|-----------|---------|----------|
| Edge TTS | edge-tts Python lib | Unlimited, no API key | Very natural, multi-language | Primary |
| Cloudflare Workers AI | TTS model endpoint | Included in free tier | Good | Fallback |

---

## 6. Functional Requirements

### 6.1 Core Features (MVP — Phase 1)

| ID | Feature | Description | Acceptance Criteria |
|----|---------|-------------|-------------------|
| FR-001 | Voice Input | Capture speech via mic, convert to text via cloud STT | >90% accuracy for clear speech in Indonesian and English |
| FR-002 | LLM Processing | Send text to cloud LLM with system prompt and context | Response within 3s (LLM only) for typical queries |
| FR-003 | Voice Output | Convert LLM response to speech, play through speakers | Natural-sounding, supports Indonesian and English |
| FR-004 | Conversation Context | Maintain history using sliding window | Last 10 exchanges retained; summarization when exceeded |
| FR-005 | API Failover | Auto-switch to fallback on error or rate limit | Failover within 2s; no crash, only brief delay |
| FR-006 | Text Input Fallback | Accept typed text when voice unavailable | Same LLM pipeline as voice input |
| FR-007 | Bilingual Support | Handle Indonesian and English seamlessly | Auto-detect language; respond in same language |

### 6.2 Enhanced Features (Phase 2)

| ID | Feature | Description | Acceptance Criteria |
|----|---------|-------------|-------------------|
| FR-101 | Wake Word | Activate via "Hey Nova" using local detection | Detected within 1s; false positive < 5% |
| FR-102 | System Control | Voice commands: volume, brightness, open apps, lock screen | At least 8 commands working reliably |
| FR-103 | Web Search | Search web and summarize results via voice | Relevant summary within 5s for factual queries |
| FR-104 | Response Caching | Cache frequent responses in local SQLite | Cache hit reduces latency by >50%; 24h invalidation |
| FR-105 | Audio Streaming | Stream TTS audio as it generates | First audio before full generation completes |

### 6.3 Future Features (Phase 3)

| ID | Feature | Description |
|----|---------|-------------|
| FR-201 | Smart Home | Home Assistant integration for IoT control |
| FR-202 | Calendar | Voice-driven Google Calendar management |
| FR-203 | Notifications | Proactive reminders and alerts via TTS |
| FR-204 | Custom Personas | Switchable AI personality profiles |
| FR-205 | Plugin System | Extensible architecture for community skills |

---

## 7. Non-Functional Requirements

| Category | Requirement | Target |
|----------|------------|--------|
| Performance | End-to-end latency | < 5 seconds (stable network) |
| Performance | Local memory consumption | < 300 MB |
| Performance | CPU during idle (wake word) | < 10% |
| Performance | CPU during processing | < 30% |
| Reliability | API failover success rate | > 95% |
| Reliability | Session uptime | > 99% (no crashes) |
| Reliability | Graceful offline degradation | Inform user; text-only via cached |
| Usability | Setup time from fresh Ubuntu | < 30 minutes |
| Usability | Voice recognition accuracy | > 90% word accuracy |
| Maintainability | Add new API provider | < 1 hour dev time |
| Maintainability | Code test coverage | > 70% for core |
| Security | API key storage | .env only, never hardcoded |
| Security | Conversation logs | Local only, no cloud |

---

## 8. API Rate Limit Analysis

| Provider | Service | Free Tier Limit | Rate Limit | Risk |
|----------|---------|----------------|-----------|------|
| Google Gemini | LLM | 1500 req/day (2.0 Flash) | 15 RPM | Low |
| Groq | LLM + STT | Varies by model | 30 RPM (LLM) | Medium |
| Cloudflare Workers AI | LLM + TTS + STT | 10K neurons/day | No per-min limit | Medium |
| Edge TTS | TTS | Unlimited | None known | Very Low |
| OpenRouter | LLM | Varies | Varies | High |
| DuckDuckGo | Web Search | Unofficial API | Undocumented | High |

### Mitigation Strategies

- **Token Conservation:** Sliding window context (last 10 exchanges) with periodic summarization. ~40–60% token reduction vs. full history.
- **Response Caching:** Common queries (time, weather, definitions) cached in local SQLite. Cache hit = zero API call.
- **Smart Routing:** Track remaining quota per provider. Route to highest-remaining-capacity provider, not just priority order.
- **Exponential Backoff:** On 429, wait 1s → 2s → 4s. Switch to fallback if wait > 3 seconds.

---

## 9. Technical Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Language | Python 3.10+ | Primary development language |
| Audio Capture | sounddevice | Microphone input |
| Audio Playback | mpv (python-mpv) | Speaker output with streaming |
| HTTP Client | httpx (async) | Async API calls with timeouts |
| TTS | edge-tts | Microsoft Edge TTS (primary) |
| Wake Word | pvporcupine | Local hotword detection |
| Database | SQLite3 | Response cache, conversation logs |
| Config | pydantic-settings | Typed environment management |
| Process Manager | systemd | Auto-start, process supervision |
| Logging | logging + rich | Structured logging |
| Testing | pytest + pytest-asyncio | Unit and integration tests |
| Linting | ruff | Code quality |

---

## 10. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Free APIs discontinued or terms changed | Medium | Critical | Multi-provider failover; abstract interfaces for easy replacement |
| Network latency > 10s | Medium | High | Response streaming; caching; timeout with feedback |
| Rate limits during heavy usage | High | Medium | Smart quota tracking; auto provider rotation |
| Poor mic quality → STT errors | Medium | Medium | Recommend external USB mic; noise filtering; VAD |
| RAM exhaustion in extended sessions | Low | High | Monitor memory; restart if threshold exceeded; limit cache |
| Wake word false positives | Medium | Low | Adjustable sensitivity; confirmation beep; push-to-talk fallback |
| Dependency conflicts | Low | Medium | Pin versions; setup script; document Python version |

---

## 11. Success Metrics

### Phase 1 (MVP)
1. Voice interaction works end-to-end within 5 seconds (stable network)
2. Failover works when primary provider is rate-limited
3. Runs 1+ hour without crash or exceeding 300MB RAM
4. Bilingual support (Indonesian + English) works
5. Total cost: $0

### Phase 2
1. Wake word detection with <5% false positive rate
2. At least 8 system commands via voice
3. Cache reduces average latency by >30% for repeated queries

---

## 12. Out of Scope

- Local/on-device LLM inference (hardware insufficient)
- GUI/desktop application (terminal + voice only)
- Multi-user support (single-user only)
- Mobile companion app
- Paid API tiers or subscriptions
- Real-time video/image processing
- Custom wake word model training

---

## 13. Glossary

| Term | Definition |
|------|-----------|
| STT | Speech-to-Text: converting spoken audio to text |
| TTS | Text-to-Speech: converting text to spoken audio |
| LLM | Large Language Model: AI for language understanding/generation |
| VAD | Voice Activity Detection: detecting speech vs. silence |
| RPM | Requests Per Minute: API rate limit measurement |
| Orchestrator | Local Python process coordinating all pipeline components |
| Failover | Automatic switch from failed/limited service to backup |
| Sliding Window | Context strategy keeping only the N most recent exchanges |
