## All Features

### ml-25m
| Feature | File | Type |
|---|---|---|
| userId | ratings | numerical |
| movieId | ratings | numerical |
| rating | ratings | numerical (1-5) |
| timestamp | ratings | numerical |
| title | movies | text |
| year | movies (extracted) | numerical |
| genres | movies | categorical multi-label |
| tag | tags | text |
| userId (tagger) | tags | numerical |
| timestamp (tag) | tags | numerical |
| imdbId | links | numerical |
| tmdbId | links | numerical |
| genome tag names | genome-tags | text |
| genome scores | genome-scores | numerical (0-1) per tag |

### TMDB API
| Feature | Type |
|---|---|
| overview / synopsis | text |
| poster URL | image |
| release date | numerical |
| runtime | numerical |
| vote average | numerical (0-10) |
| vote count | numerical |
| popularity score | numerical |
| original language | categorical |
| tmdb genres | categorical multi-label |
| cast (top 5) | text / multi-hot |
| director | text / categorical |