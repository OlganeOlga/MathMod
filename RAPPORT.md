Uppgift 1: Beskriv data 
Introducera den data som valts och beskriv vad den visar och varifrån den kommer. Cirka 250 ord 
(halv A4). Var tydliga med vad de olika variablerna beskriver och i vilken enhet de är i. Det kan vara 
en god idé att ha en mindre tabell med ett urval från datan för att lättare beskriva mätvärdena.  
Det ska också finnas en visuell representation av hur datamängden ser ut, samt tillhörande figurtext 
med förklaringar till vad som visas och om det finns några konstigheter (till exempel outliers i datan). 
Visualiseringen görs med lämplig graf, t.ex. stapeldiagram, linjediagram, scatterplot, cirkeldiagram 
etc. Obs! Glöm inte att ange enheter på axlarna! 
### RESULTAT
Dessa tabeller skapas med filen `get_dynam_data/prepere_data.py`
## TEMPERATUR per timme under sista tre dagar från tre stationer:
## Data Table
|                     |   Halmstad flygplats(°C) |   Uppsala Flygplats(°C) |   Umeå Flygplats(°C) |
|:--------------------|-------------------------:|------------------------:|---------------------:|
| 2024-12-15 08:00:00 |                      3   |                    -1.7 |                -11.1 |
| 2024-12-15 09:00:00 |                      4.6 |                    -1.7 |                -12.4 |
| 2024-12-15 10:00:00 |                      5.3 |                    -1.8 |                -14.6 |
| 2024-12-15 11:00:00 |                      5.3 |                    -2   |                -13.2 |
.......
| 2024-12-17 06:00:00 |                      6.3 |                     2.3 |                 -4.7 |
| 2024-12-17 07:00:00 |                      6.3 |                     2.2 |                 -4.8 |
| 2024-12-17 08:00:00 |                      5.9 |                     2.1 |                 -5   |
| 2024-12-17 09:00:00 |                      6.5 |                     2.3 |                 -5.4 |
| 2024-12-17 10:00:00 |                      7.1 |                     2.6 |                 -5.2 |
| 2024-12-17 11:00:00 |                      7   |                     2.8 |                 -5.7 |
| 2024-12-17 12:00:00 |                      7   |                     2.9 |                 -8.4 |


## LUFTFUKTIGHET per timme från tre stationer

## Data Table
|                     |   Halmstad flygplats(%) |   Uppsala Flygplats(%) |   Umeå Flygplats(%) |
|:--------------------|------------------------:|-----------------------:|--------------------:|
| 2024-12-15 08:00:00 |                      89 |                     84 |                  89 |
| 2024-12-15 09:00:00 |                      88 |                     84 |                  88 |
| 2024-12-15 10:00:00 |                      86 |                     84 |                  85 |
| 2024-12-15 11:00:00 |                      86 |                     82 |                  88 |
| 2024-12-15 12:00:00 |                      90 |                     81 |                  86 |
| 2024-12-15 13:00:00 |                      96 |                     82 |                  88 |
| 2024-12-15 14:00:00 |                      97 |                     85 |                  86 |
| 2024-12-15 15:00:00 |                      98 |                     84 |                  87 |
| 2024-12-15 16:00:00 |                      98 |                     86 |                  88 |
| 2024-12-15 17:00:00 |                      98 |                     99 |                  90 |
| 2024-12-15 18:00:00 |                      95 |                    100 |                  92 |
.......
| 2024-12-17 08:00:00 |                      92 |                     62 |                  90 |
| 2024-12-17 09:00:00 |                      90 |                     61 |                  89 |
| 2024-12-17 10:00:00 |                      90 |                     59 |                  85 |
| 2024-12-17 11:00:00 |                      89 |                     58 |                  85 |
| 2024-12-17 12:00:00 |                      91 |                     57 |                  87 |


Jag tittar om det finns missade data:

```
import get_dynam_data.prepere_data as p_d

# missade data för temperatur
data = p_d.data_from_file(param=1)
three_days = p_d.extract_for_statistics(data=data)
df, stats = p_d.data_describe(three_days)

#get missing data:
missing_data = df.isna()
#summary of missing data: 
missing_summary = df.isna().sum()

p_d.append_to_markdown(missing_summary)
data = p_d.data_from_file(param=6)
three_days = p_d.extract_for_statistics(data=data)
df, stats = p_d.data_describe(three_days)

# missade data för LUFTFUKTIGHET
missing_data = df.isna()
#summary of missing data: 
missing_summary = df.isna().sum()
print(missing_summary)
p_d.append_to_markdown(missing_summary) 
```
----
### Missade data för TEMPERATUR             ## Missade data för LUFTFUKTIGHET
|                    |   0 |                |                    |   0 |
|:-------------------|----:|                |:-------------------|----:|
| Halmstad flygplats |   0 |                | Halmstad flygplats |   0 |
| Umeå Flygplats     |   0 |                | Umeå Flygplats     |   0 |
| Uppsala Flygplats  |   0 |                | Uppsala Flygplats  |   0 |


Det verkar att inga data missade för detta tids interval.
Dessa tabeller är skapade med samma fil som första två:

## Beskrivande statistik TEMPERATUR
    mäts varje timme 
|       |   Halmstad flygplats(°C) |   Uppsala Flygplats(°C) |   Umeå Flygplats(°C) |
|:------|-------------------------:|------------------------:|---------------------:|
| count |                    53    |                   53    |                53    |
| mean  |                     6.99 |                    1.71 |                -9.07 |
| std   |                     1.14 |                    2.33 |                 4.58 |
| min   |                     3    |                   -2.5  |               -17.2  |
| 25%   |                     6.5  |                    1.2  |               -13.9  |
| 50%   |                     7.2  |                    2.3  |                -8.4  |
| 75%   |                     7.6  |                    2.9  |                -5.2  |
| max   |                     8.9  |                    6.6  |                -1.3  |

Medelvärde i stationer Halmstad Flugplats och Upsala Flugplats är närmare medianen, som säger att de ssa data 
närmare normafördelning än data från Umeå Flugplats

## Beskrivande statistik RELATIVT LUFTFUKTIGHET
    mäts varje timme 
|       |   Halmstad flygplats(%) |   Uppsala Flygplats(%) |   Umeå Flygplats(%) |
|:------|------------------------:|-----------------------:|--------------------:|
| count |                   53    |                  53    |               53    |
| mean  |                   89.32 |                  78.49 |               89.19 |
| std   |                    5.95 |                  13.36 |                3.16 |
| min   |                   75    |                  57    |               84    |
| 25%   |                   86    |                  66    |               87    |
| 50%   |                   91    |                  81    |               89    |
| 75%   |                   93    |                  85    |               91    |
| max   |                   98    |                 100    |               96    |


![Temperatur frekvenser](img/frekvenser/TEMPERATUR_combined.png)

Dessa plottar visar tydligt att temperatur spridning är inte normal färdelad, desuttom normalfordelningstest 
visar väldignt litet sannolikhet för normalfördelning

![Luftfuktighet frekvenser](img/frekvenser/LUFTFUKTIGHET_combined.png)

Samma resultat visas gällande lurftfuktigheter
