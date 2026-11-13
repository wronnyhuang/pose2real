**The case for more budget.** $11.33 of the $30 was spent. The RunPod balance is now about $7.50, and another $5–10 of runs would need a top-up to keep the live demo's pod safe. Landmark fidelity is good and still improving; lip sync is not there. What would move it, in order:

- **A proper lip-sync loss.** The quick SyncNet-loss runs either cheated or hurt. Wav2Lip-style training uses a large batch of mouth-only crops and its own sync expert trained on the same data, kept separate from the SyncNet used for scoring. That is a 3–4 hour job on one RTX 5090, about $3.
- **Mouth interior from audio.** Landmarks do not say whether teeth or tongue show. Feed the audio tokens straight into the UNet's cross-attention for the mouth region, so the image model, not only the landmark encoder, hears the sound. About $2.
- **More people.** Only 33 identities train the model. The full Seamless set (thousands of people) or CANDOR would let the reference photo actually set the face for a new person. Mostly host CPU time for landmarks; training about $4.
- **A temporal backbone.** Wan VACE 1.3B or an AnimateDiff motion module, conditioned the same way, for teeth and lip motion that stay coherent across frames. Wan is not real-time; about $5 to test.
- **Real-time path.** 25.6 ms on a 5090 is already above 30 fps. A TAESD decoder would cut the 19 ms VAE step to about 3 ms.
