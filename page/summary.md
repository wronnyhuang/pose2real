**Your design works.** Face landmarks, body skeleton and audio go through a trainable encoder; the mixed embedding is added to the starting noise of one-step SD-Turbo; a reference photo fixes the look. No render is used.

- **Landmark fidelity is good.** Best model: 1.6% face, 1.7% mouth, 1.7% skeleton error. The render→photoreal fast model is at 2.1 / 2.0 / 2.5; the video-compression noise floor is 0.5 / 0.5 / 0.7.
- **Lip sync is not there yet.** SyncNet confidence 3.47 (distance 9.84). Real video scores 6.16 (7.83); the render→photoreal fast model 3.11 (9.94).
- **Realism 5.2 out of 10** against 8.1 for real video and 5.4 for the render→photoreal fast model. The judge never took it for real footage.
- **Speed 25.6 ms per frame (39 fps)** at 384 px on an RTX 5090.
- **Spend $11.33** of the $30 GPU budget. All pods are terminated.

The best numbers are for people the model saw other footage of (16–32 s each, never the test clips). For people it never saw, landmark error is 1.8 / 1.9 / 2.3 and lip sync 3.41.
