# AI Surveillance System - Implementation Plan

**Team:** 4 Members | **Timeline:** 4 Weeks | **Target:** 16GB RAM, Intel UHD Graphics

---

## Project Overview

Build a real-time AI-powered surveillance system that detects suspicious activities using:
- **YOLOv8** for object detection (weapons, bags, people, vehicles)
- **DeepSORT** for multi-object tracking
- **VL-JEPA2** (`facebook/vjepa2-vitl-fpc64-256`, 0.7B params) for video-native behavior analysis (fighting, running, loitering, crowding, weapon display, falling)
- **Real-time alerts** via Email/SMS/Webhook
- **Web dashboard** for monitoring

> **Behavior Backend Decision (2026-03-29):** VL-JEPA2 selected over MediaPipe.
> MediaPipe (pose heuristics, single-frame) was discarded — it cannot model temporal context, multi-person scenes, or learned behavior classes.
> VL-JEPA2 is a video-native model trained to understand action in time, which is the correct architectural fit for surveillance.

---

## Architecture Diagram

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│ CCTV/Webcam │ ──→ │ Frame Processing │ ──→ │ Alert System│
└─────────────┘     └──────────────────┘     └──────┬──────┘
       │                                            │
       │                    ┌───────────────────────┼───────────────┐
       │                    │                       │               │
       ▼                    ▼                       ▼               ▼
┌─────────────────┐  ┌──────────────────┐   ┌───────────┐   ┌─────────────┐
│ YOLOv8 Detection│  │ VL-JEPA2         │   │ Database  │   │ SMS/Email   │
│ (Person/Vehicle) │  │ 0.7B Video Model │   │ (SQLite)  │   │ Webhook     │
└────────┬────────  │ Behavior Analysis│   └───────────┘   └─────────────┘
         │           │ (8 action classes)│
         ▼           └──────┬──────────┘
┌─────────────────┐         │
│ DeepSORT Tracking│        │
│ (ID Assignment)  │────────┘
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Web Dashboard   │
│ (Real-time UI)  │
└─────────────────┘
```

---

## Team Roles & Responsibilities

| Member | Role | Primary Tasks | Secondary Tasks |
|--------|------|---------------|-----------------|
| **Shivam** | Core Development | YOLOv8 integration, DeepSORT tracking, Main pipeline | Behavior analysis integration |
| **Sumedh** | Dashboard & UI | Streamlit monitoring dashboard, Team dashboard | Database schema design |
| **Snehasich** | Testing & QA | Unit tests, Integration tests, Performance benchmarks | Documentation review |
| **Suhas** | Documentation & DevOps | Setup docs, Docker containerization, CI/CD | Requirements management |

---

## Week-by-Week Breakdown

### Week 1: Foundation & Core Detection

#### Shivam - Core Development
| Task | Status | Priority | Estimated Hours |
|------|--------|----------|-----------------|
| Set up YOLOv8 ONNX export | ✅ Done | High | 4h |
| Implement YOLOv8Detector class | ✅ Done | High | 6h |
| Configure confidence/IoU thresholds | ✅ Done | Medium | 2h |
| Test detection on sample images | ⬜ Pending | High | 3h |

**Deliverable:** Working object detection with bounding boxes

#### Sumedh - Dashboard & UI
| Task | Status | Priority | Estimated Hours |
|------|--------|----------|-----------------|
| Set up Streamlit environment | ✅ Done | High | 2h |
| Design dashboard layout | ✅ Done | Medium | 4h |
| Create real-time event display component | ✅ Done | High | 6h |
| Implement severity color coding | ✅ Done | Medium | 2h |

**Deliverable:** Basic dashboard showing mock events

#### Snehasich - Testing & QA
| Task | Status | Priority | Estimated Hours |
|------|--------|----------|-----------------|
| Set up pytest framework | ✅ Done | High | 2h |
| Write YOLO detector tests | ⬜ Pending | High | 4h |
| Create test fixtures and mock data | ⬜ Pending | Medium | 4h |
| Document test coverage baseline | ⬜ Pending | Low | 2h |

**Deliverable:** Test framework + 20+ unit tests

#### Suhas - Documentation & DevOps
| Task | Status | Priority | Estimated Hours |
|------|--------|----------|-----------------|
| Create SETUP.md guide | ✅ Done | High | 4h |
| Document requirements.txt | ✅ Done | High | 2h |
| Research VL-JEPA2 optimization for Intel UHD | ✅ Done | Medium | 4h |
| Create README.md structure | ✅ Done | Medium | 3h |

**Deliverable:** Complete setup documentation

---

### Week 2: Tracking & Behavior Analysis

#### Shivam - Core Development
| Task | Status | Priority | Estimated Hours |
|------|--------|----------|-----------------|
| Implement DeepSORT tracker | ✅ Done | High | 8h |
| Add Kalman filter for prediction | ✅ Done | High | 4h |
| Implement Hungarian matching | ✅ Done | High | 4h |
| Integrate VL-JEPA2 behavior analysis | ✅ Done | High | 6h |
| Train/attach classification head for VL-JEPA2 embeddings | ⬜ Pending | High | 8h |

**Deliverable:** Multi-object tracking + VL-JEPA2 behavior pipeline with real classifier head

#### Sumedh - Dashboard & UI
| Task | Status | Priority | Estimated Hours |
|------|--------|----------|-----------------|
| Connect dashboard to live database | ⬜ Pending | High | 4h |
| Add statistics charts (by type, severity) | ⬜ Pending | Medium | 4h |
| Implement event filtering UI | ⬜ Pending | Medium | 3h |
| Create team progress dashboard | ⬜ Pending | Low | 4h |

**Deliverable:** Fully connected monitoring dashboard

#### Snehasich - Testing & QA
| Task | Status | Priority | Estimated Hours |
|------|--------|----------|-----------------|
| Write DeepSORT tracker tests | ⬜ Pending | High | 4h |
| Test Kalman filter calculations | ⬜ Pending | Medium | 3h |
| Integration tests: detection + tracking | ⬜ Pending | High | 4h |
| Performance benchmarking setup | ⬜ Pending | Medium | 3h |

**Deliverable:** Tracking tests + integration test suite

#### Suhas - Documentation & DevOps
| Task | Status | Priority | Estimated Hours |
|------|--------|----------|-----------------|
| Document API endpoints | ⬜ Pending | Medium | 4h |
| Create Dockerfile | ⬜ Pending | High | 4h |
| Set up GitHub Actions CI | ⬜ Pending | Medium | 4h |
| Write troubleshooting guide | ⬜ Pending | Medium | 3h |

**Deliverable:** Docker container + CI pipeline

---

### Week 3: Alerting & Database

#### Shivam - Core Development
| Task | Status | Priority | Estimated Hours |
|------|--------|----------|-----------------|
| Implement RuleEngine class | ⬜ Pending | High | 4h |
| Create detection rules (weapon, fight, loitering) | ⬜ Pending | High | 4h |
| Add cooldown/rate limiting | ⬜ Pending | Medium | 2h |
| Integrate behavior analysis | ⬜ Pending | High | 6h |

**Deliverable:** Rule-based alert triggering system

#### Sumedh - Dashboard & UI
| Task | Status | Priority | Estimated Hours |
|------|--------|----------|-----------------|
| Add alert notification display | ⬜ Pending | High | 3h |
| Implement real-time refresh | ⬜ Pending | High | 3h |
| Create database cleanup UI | ⬜ Pending | Low | 2h |
| Add export functionality (CSV) | ⬜ Pending | Low | 3h |

**Deliverable:** Alert notifications + data export

#### Snehasich - Testing & QA
| Task | Status | Priority | Estimated Hours |
|------|--------|----------|-----------------|
| Write alert engine tests | ⬜ Pending | High | 4h |
| Test rule evaluation logic | ⬜ Pending | High | 3h |
| Database CRUD operation tests | ⬜ Pending | Medium | 3h |
| End-to-end pipeline tests | ⬜ Pending | High | 4h |

**Deliverable:** Alert tests + E2E test coverage

#### Suhas - Documentation & DevOps
| Task | Status | Priority | Estimated Hours |
|------|--------|----------|-----------------|
| Document database schema | ⬜ Pending | Medium | 3h |
| Create alert configuration guide | ⬜ Pending | Medium | 3h |
| Set up automated testing in CI | ⬜ Pending | High | 4h |
| Performance optimization docs | ⬜ Pending | Medium | 4h |

**Deliverable:** Schema docs + automated testing

---

### Week 4: Integration & Polish

#### Shivam - Core Development
| Task | Status | Priority | Estimated Hours |
|------|--------|----------|-----------------|
| Main pipeline integration | ⬜ Pending | High | 6h |
| Performance optimization (16GB RAM) | ⬜ Pending | High | 4h |
| Memory management improvements | ⬜ Pending | Medium | 3h |
| Bug fixes from QA feedback | ⬜ Pending | High | 4h |

**Deliverable:** Production-ready main.py

#### Sumedh - Dashboard & UI
| Task | Status | Priority | Estimated Hours |
|------|--------|----------|-----------------|
| Dashboard performance optimization | ⬜ Pending | Medium | 3h |
| Add keyboard shortcuts | ⬜ Pending | Low | 2h |
| Polish UI/UX | ⬜ Pending | Medium | 4h |
| Create user guide | ⬜ Pending | Medium | 3h |

**Deliverable:** Polished, user-friendly dashboard

#### Snehasich - Testing & QA
| Task | Status | Priority | Estimated Hours |
|------|--------|----------|-----------------|
| Full system regression testing | ⬜ Pending | High | 6h |
| Performance benchmarking | ⬜ Pending | High | 4h |
| Bug verification and reporting | ⬜ Pending | High | 4h |
| Test documentation | ⬜ Pending | Medium | 2h |

**Deliverable:** Test report + bug list

#### Suhas - Documentation & DevOps
| Task | Status | Priority | Estimated Hours |
|------|--------|----------|-----------------|
| Final README update | ⬜ Pending | High | 3h |
| Release notes preparation | ⬜ Pending | Medium | 3h |
| Deployment guide | ⬜ Pending | High | 4h |
| Create demo video/screenshots | ⬜ Pending | Low | 4h |

**Deliverable:** Complete documentation package

---

## Technical Specifications (16GB RAM + VL-JEPA2 Optimized)

### VL-JEPA2 Model Specs
| Property | Value |
|----------|-------|
| Model | `facebook/vjepa2-vitl-fpc64-256` |
| Parameters | 0.7B (ViT-Large backbone) |
| Architecture | Vision-Language Joint Embedding Predictive Architecture |
| Training | Video clips — temporal context built-in |
| Behavior classes | normal, fighting, running, loitering, crowding, weapon_display, falling, suspicious_activity |
| Input | 224×224 frames (16-frame clip buffer) |
| Quantization | INT8 via bitsandbytes (required for 16GB RAM) |
| Model size on disk | ~2.8 GB (float16) / ~1.4 GB (INT8) |

### Memory Budget (16GB RAM with VL-JEPA2 INT8)
| Component | Allocation |
|-----------|------------|
| VL-JEPA2 Model (INT8) | ~1.5 GB |
| YOLOv8n Model | 500 MB |
| DeepSORT Tracker | 300 MB |
| Frame Buffer (16 frames) | 800 MB |
| Dashboard | 400 MB |
| System Overhead | 2 GB |
| **Total Reserved** | **~5.5 GB** |

### Performance Targets (VL-JEPA2 Backend)
| Metric | CPU-only | With OpenVINO | With CUDA GPU |
|--------|----------|----------------|---------------|
| FPS | 5-8 FPS | 8-12 FPS | 15-20 FPS |
| Latency | ~200ms | ~120ms | ~60ms |
| Behavior classes | 8 | 8 | 8 |
| Temporal window | 16 frames | 16 frames | 16 frames |

### Optimization Strategies
1. **INT8 Quantization:** `bitsandbytes` reduces model from 2.8GB → 1.4GB
2. **Frame Skipping:** `frame_skip: 2` — feed every 3rd frame to VL-JEPA2
3. **Resolution:** 224×224 input (not 256) for 16GB RAM
4. **OpenVINO:** Enable for Intel UHD Graphics with `use_openvino: true`
5. **Lazy Loading:** VL-JEPA2 loads on first `analyze()` call, not at startup
6. **Clip Buffer:** 16-frame deque — VL-JEPA2 analyses temporal patterns, not single frames
7. **Garbage Collection:** Every 100 frames (`gc_interval: 100` in config)

---

## Risk Management

| Risk | Impact | Mitigation |
|------|--------|------------|
| Out of memory errors | High | Frame skip, reduced batch size |
| Low FPS on Intel UHD | Medium | OpenVINO, smaller model |
| Camera connection fails | Medium | Support video file fallback |
| MediaPipe too slow | Low | Use pose-only mode |

---

## Milestone Checklist

### Week 1 Milestone
- [ ] YOLOv8 detecting objects in real-time
- [ ] Dashboard showing events
- [ ] 20+ unit tests passing
- [ ] SETUP.md complete

### Week 2 Milestone
- [ ] DeepSORT tracking with consistent IDs
- [ ] Dashboard connected to database
- [ ] Integration tests passing
- [ ] Docker container working

### Week 3 Milestone
- [ ] Alert rules triggering correctly
- [ ] Email/SMS alerts functional
- [ ] Database storing all events
- [ ] CI pipeline running tests

### Week 4 Milestone
- [ ] Full pipeline running end-to-end
- [ ] Performance meets targets (10+ FPS)
- [ ] All tests passing (>80% coverage)
- [ ] Documentation complete

---

## Daily Standup Template

```markdown
## [Date] - [Name]

### Yesterday
- Task 1: Status
- Task 2: Status

### Today
- Task 1: Planned
- Task 2: Planned

### Blockers
- [Any issues blocking progress]
```

---

## Git Workflow

```bash
# Main branches
main          # Production-ready code
develop       # Integration branch

# Feature branches (per person)
feature/shivam-detection
feature/sumedh-dashboard
feature/snehasich-tests
feature/suhas-docs

# Workflow
git checkout develop
git checkout -b feature/your-feature
# ... make changes ...
git commit -m "feat: description"
git push origin feature/your-feature
# Create PR on GitHub
```

---

## Communication Plan

| Channel | Purpose |
|---------|---------|
| WhatsApp | Quick updates, blockers |
| GitHub | Code reviews, PRs, issues |
| Google Meet | Weekly sync (Monday 10 AM) |
| Shared Doc | Meeting notes, decisions |

---

## Final Deliverables

1. **Working Application**
   - `main.py` - Surveillance pipeline
   - `dashboard/app.py` - Monitoring UI
   - `team_dashboard.py` - Progress tracking

2. **Documentation**
   - README.md - Project overview
   - SETUP.md - Installation guide
   - IMPLEMENTATION_PLAN.md - This document

3. **Tests**
   - 50+ unit tests
   - Integration test suite
   - Performance benchmarks

4. **Deployment**
   - Docker container
   - CI/CD pipeline
   - Release package
