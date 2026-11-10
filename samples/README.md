# samples/

Intentionally empty. See `docs/TROUBLESHOOTING.md` → "Why there are no real
face images in `samples/`" for the full explanation — in short, this
project avoids bundling any real person's photo without clear, traceable
consent, even widely-used tutorial images.

## Quickest way to try it

Drop any photo you have the rights to use directly into this folder and
point `POST /api/analyze/image` at it.

## Running the real accuracy evaluation

`backend/tests/test_evaluation_real_faces.py` is a full true-positive /
false-positive evaluation suite — it's written and ready, but every test in
it is auto-skipped until you add photos in this exact layout (use people
who've actually agreed to be in a face-recognition test fixture):

```
samples/eval/
  identities/
    <person-name>/
      enroll.jpg        # the photo that gets enrolled
      query_same.jpg    # a DIFFERENT photo of the SAME person — must be
                         # recognized as a true positive
  distractors/
    *.jpg                # photos of people who are NOT enrolled — must
                          # come back "Unknown", never a false match
  no_face/
    *.jpg                # genuine photos with no face in them at all
```

Once that's in place:

```powershell
cd backend
python -m pytest tests/test_evaluation_real_faces.py -v -s
```

The `-s` flag shows the printed match-distance summary for each identity,
not just pass/fail — useful for tuning
`settings.recognition_distance_threshold` in `backend/app/core/config.py`
against your own enrolled population, which `docs/MODEL_CARDS.md` already
flags as something the shipped default (`0.68`) was never verified against.

## Running the age-estimation evaluation

Same idea, for `backend/tests/test_age_evaluation_real_faces.py`: skipped
until you add labeled photos (people who've agreed to be in an age-estimation
test fixture, ideally spanning a real age range including some clearly young
faces) in this layout:

```
samples/eval/
  ages/
    <any-label>/
      photo.jpg          # or .jpeg / .png
      true_age.txt       # a single integer: the person's real age in that photo
```

Then:

```powershell
cd backend
python -m pytest tests/test_age_evaluation_real_faces.py -v -s
```

This checks three things: the estimates are roughly in the right neighborhood
(loose on purpose — it's flagging "badly wrong," not certifying a precise
error rate), the model isn't silently falling back to one hardcoded number
for every photo, and — the specific regression this evaluation exists to
catch — any photo labeled with a true age under 13 doesn't come back
estimated at 20 or older. That last check is what would have caught the
original adult-defaulting bug in `age_estimation.py` before it shipped.
