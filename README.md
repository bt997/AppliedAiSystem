# 🎵 Music Recommender Simulation

## Project Summary

The project I chose to expand on was the music recommender system.

Previously, this project simulated a music recommender system using a catalog of 18 songs. Given a user's preferred genre, mood, and/or artist, the system scores every song and returns the top matches. Genre matches add 2 points, mood matches add 3, and artist matches add 5. Songs with equal scores are ranked by danceability as a tiebreaker. The goal is to explore how simple scoring rules can drive recommendations and where they fall short.

The goal of the expansion was to integrate Retrieval-Augmented-Generation(RAG) into the song fetching process.

---

## How The System Works

Looking at all the attributes in songs.csv, I want my recommender to prioritize mood and genre if the user isn't looking for a specific title or artist. I wouldn't worry too much about having the user input energy, tempo_bpm, valence, danceability or acousticness because those look like arbitrary numbers that I wouldn't be able to answer if someone asked me what song I was looking for.

```mermaid
flowchart TD
    A([User Input]) --> B{search_title\nprovided?}

    B -- Yes --> C{Any song title\ncontains it?}
    C -- No --> D([Return: Song not found])
    C -- Yes --> E([Return top match only])

    B -- No --> F[Score every song]
    F --> G[genre contains match? +2]
    G --> H[mood match? +3]
    H --> I[artist match? +5]
    I --> J[Sort by score descending]
    J --> K{Top score == 0?}
    K -- Yes --> N([Warn: No good match found])
    K -- No --> L{Tie in score?}
    L -- Yes --> O[Break tie by danceability\nnotify user]
    L -- No --> M([Return top k songs])
    O --> M
```

The biases to be expected are exact string matching and having a small catalog, which can be fixed if more songs are added to it.

![Recommendations output](recommendations.png)
---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows

2. Install dependencies and Google API Key

Go to Google AI Studio and create a new project and key.

For Mac:
```bash
echo 'export GEMINI_API_KEY=your-key-here' >> ~/.zshrc
```

```bash
source ~/.zshrc
```

```bash
python -m pip install -r requirements.txt
```

3. Run the app:

```bash
python src/main.py
```

### Running Tests

Run the starter tests with:

```bash
pytest tests/ -v
```

---

## Experiments You Tried

Use this section to document the experiments you ran. For example:

- What happened when you changed the weight on genre from 2.0 to 0.5
- What happened when you added tempo or valence to the score
- How did your system behave for different types of users

---

## Limitations and Risks

Summarize some limitations of your recommender.

Examples:

- It only works on a tiny catalog
- It does not understand lyrics or language
- It might over favor one genre or mood

You will go deeper on this in your model card.

![Edge case profiles](edgecase.png)
![Edge case results](edgecases.png)

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

Write 1 to 2 paragraphs here about what you learned:

- about how recommenders turn data into predictions
- about where bias or unfairness could show up in systems like this


---

## 7. `model_card_template.md`

Combines reflection and model card framing from the Module 3 guidance. :contentReference[oaicite:2]{index=2}  

```markdown
# 🎧 Model Card - Music Recommender Simulation

## 1. Model Name

Give your recommender a name, for example:

> VibeFinder 1.0

---

## 2. Intended Use

- What is this system trying to do
- Who is it for

Example:

> This model suggests 3 to 5 songs from a small catalog based on a user's preferred genre, mood, and energy level. It is for classroom exploration only, not for real users.

---

## 3. How It Works (Short Explanation)

Describe your scoring logic in plain language.

- What features of each song does it consider
- What information about the user does it use
- How does it turn those into a number

Try to avoid code in this section, treat it like an explanation to a non programmer.

Refer to model_card.md

---

## 4. Data

Describe your dataset.

- How many songs are in `data/songs.csv`
- Did you add or remove any songs
- What kinds of genres or moods are represented
- Whose taste does this data mostly reflect

Refer to model_card.md

---

## 5. Strengths

Where does your recommender work well

You can think about:
- Situations where the top results "felt right"
- Particular user profiles it served well
- Simplicity or transparency benefits

Refer to model_card.md

---

## 6. Limitations and Bias

Where does your recommender struggle

Some prompts:
- Does it ignore some genres or moods
- Does it treat all users as if they have the same taste shape
- Is it biased toward high energy or one genre by default
- How could this be unfair if used in a real product

Refer to model_card.md

---

## 7. Evaluation

How did you check your system

Examples:
- You tried multiple user profiles and wrote down whether the results matched your expectations
- You compared your simulation to what a real app like Spotify or YouTube tends to recommend
- You wrote tests for your scoring logic

You do not need a numeric metric, but if you used one, explain what it measures.

Refer to model_card.md

---

## 8. Future Work

If you had more time, how would you improve this recommender

Examples:

- Add support for multiple users and "group vibe" recommendations
- Balance diversity of songs instead of always picking the closest match
- Use more features, like tempo ranges or lyric themes

Refer to model_card.md

---

## 9. Personal Reflection

A few sentences about what you learned:

- What surprised you about how your system behaved
- How did building this change how you think about real music recommenders
- Where do you think human judgment still matters, even if the model seems "smart"

Refer to model_card.md

