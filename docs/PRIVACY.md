# Privacy, Consent, and Data Retention

This document exists because a face intelligence system handles biometric
data, and biometric data deserves more care than a typical CRUD app's data.

## What this system will never do

- **Never identify a stranger.** Recognition can only ever return a name
  that was explicitly enrolled through `POST /api/faces/enroll`, or
  `"Unknown"`. There is no code path that searches the internet, a social
  network, or any external database to put a name to a face.
- **Never enroll silently.** `recognition.enroll_face()` raises
  `PermissionError` unless the caller explicitly passes
  `consent_confirmed=True`. Running detection, analysis, or tracking on an
  image or video never enrolls anyone as a side effect — enrollment is a
  separate, deliberate API call.
- **Never force a match.** If the closest enrolled identity is still
  farther than the configured threshold, the result is `"Unknown"` — the
  system does not pick "the closest person anyway".

## What "consent_confirmed=True" actually means (and doesn't)

The backend **cannot** verify real-world informed consent — it can only
refuse to proceed without the caller *asserting* it. The responsibility for
actually collecting consent belongs to whatever sits in front of this API
(the frontend's enrollment flow, or you, if you're calling the API
directly). A production deployment should show the person being enrolled a
clear explanation of what's being stored and why, and only then send
`consent_confirmed=True`. Treat this flag as a contract your UI must honor,
not as a feature that does consent-collection for you.

## What gets stored, and where

| Data | Stored? | Location | Notes |
|---|---|---|---|
| Enrolled face **embedding** (512 numbers) | Yes | `data/embeddings/index.json` | What matching actually compares against. |
| Enrolled face **image** | **No, by default** | `data/gallery/<id>.jpg` | Only written if `settings.retain_enrollment_images = True`. Off by default per the master spec's "store embeddings rather than retaining original enrollment images when practical". |
| Uploaded images/videos for one-off analysis | No | `data/tmp_uploads/` | Written to disk only for the duration of request processing, then deleted via the validation layer's cleanup (see `security/validation.py`). |
| Anything from a webcam stream | Not implemented yet | — | See README "Next Phase" — this needs explicit, visible permission UI before it's built, not just a backend endpoint. |

An embedding is **not reversible back into a photo** by this project's code
— ArcFace is a one-way feature extractor — but that does not make it
anonymous data in a legal sense in every jurisdiction; biometric-specific
regulations (e.g. BIPA in Illinois, GDPR's "special category data" in the
EU/UK) generally still treat a face embedding as biometric data requiring
its own legal basis for processing. This project does not implement
jurisdiction-specific compliance (consent-receipt logging, data-subject
export/delete-on-request workflows beyond the basic `DELETE
/api/faces/{id}`, retention-period enforcement) — treat `DELETE
/api/faces/{id}` as a useful building block, not a complete compliance
story.

## Removing someone

`DELETE /api/faces/{face_id}` removes both the embedding entry and the
retained image file, if one exists. There is currently no "delete by name"
convenience (a name is not guaranteed unique) — look up the `face_id` via
`GET /api/faces` first.

## Expression and age: what they are not evidence of

A facial-expression label is a classification of visual appearance, not a
measurement of what someone is feeling, thinking, or intending. An age
estimate is a visual guess, not a verified age and never a substitute for
an ID check. Both are rendered with explicit uncertainty/limitations notes
in every API response (see `expressions.py` and `age_estimation.py`) rather
than as bare numbers — please keep it that way if you build a UI on top of
this API; stripping the caveats out of a dashboard is how a "rough visual
estimate" quietly turns into something people start trusting as fact.
