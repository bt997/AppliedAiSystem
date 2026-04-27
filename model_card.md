# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

Give your model a short, descriptive name.  
Example: **Claude Music**  

---

## 2. Intended Use  

Describe what your recommender is designed to do and who it is for. 

Prompts:  

- What kind of recommendations does it generate  
- What assumptions does it make about the user  
- Is this for real users or classroom exploration  

It generates song recommendations based on a user's preferences. It is for classroom exploration, not real users yet.
---

## 3. How the Model Works  

Explain your scoring approach in simple language.  

Prompts:  

- What features of each song are used (genre, energy, mood, etc.)  
- What user preferences are considered  
- How does the model turn those into a score  
- What changes did you make from the starter logic  

Avoid code here. Pretend you are explaining the idea to a friend who does not program.

Genre, mood and artist are used in the scoring of songs. I avoided the use of energy, tempo_bpm, valence and acousticness. Danceability is only used as a tiebreaker.

The model compares user preference to each song attribute one at a time. Matching genres will add 2 points, matching moods will add 3 points and matching artists will add 5. If songs end up with the same score, the higher dancability song will be ranked higher.
---

## 4. Data  

Describe the dataset the model uses.  

Prompts:  

- How many songs are in the catalog  
- What genres or moods are represented  
- Did you add or remove data  
- Are there parts of musical taste missing in the dataset  

There are 18 songs in the catalog. The genres represented are: pop, lofi, rock, ambient, jazz, synthwave, indie pop, classical, latin, r&b, metal, country, soul, hip-hop and folk. 
The moods represented are happy, chill, intense, relaxed, moody, focused, sad, euphoric, romantic, angry, nostalgic, melancholic, energetic and calm.
I added songs to the sample catalog.
There are for sure a lot of musical taste missing in the dataset because there are just so many types of music out there.
---

## 5. Strengths  

Where does your system seem to work well  

Prompts:  

- User types for which it gives reasonable results  
- Any patterns you think your scoring captures correctly  
- Cases where the recommendations matched your intuition  

Since lofi and chill are matching genre and moods, those recommendations come out as expected.
---

## 6. Limitations and Bias 

Where the system struggles or behaves unfairly. 

Prompts:  

- Features it does not consider  
- Genres or moods that are underrepresented  
- Cases where the system overfits to one preference  
- Ways the scoring might unintentionally favor some users  

Edge cases are shown in the screenshots on the README.md like an empty user profile, danceability ties and partial profiles.

---

## 7. Evaluation  

How you checked whether the recommender behaved as expected. 

Prompts:  

- Which user profiles you tested  
- What you looked for in the recommendations  
- What surprised you  
- Any simple tests or comparisons you ran  

No need for numeric metrics unless you created some.

I was surprised by how quickly an algorithm recipe was created but how different it was from what I had in mind. I had to reprompt Claude to give it more of an idea of what I wanted.
---

## 8. Future Work  

Ideas for how you would improve the model next.  

Prompts:  

- Additional features or preferences  
- Better ways to explain recommendations  
- Improving diversity among the top results  
- Handling more complex user tastes  

There could always be more genres of music as mentioned before since there are so many in the world.
---

## 9. Personal Reflection  

A few sentences about your experience.  

Prompts:  

- What you learned about recommender systems  
- Something unexpected or interesting you discovered  
- How this changed the way you think about music recommendation apps  

To all music recommendation apps, we are just numbers or keywords. I already noticed before this project that my Spotify radio had so many songs I already listened to or very similar playlists even though I was comparing two different song radios.
