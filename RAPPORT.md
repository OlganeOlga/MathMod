# PROJECT RAPPRORT for kurs Matematisk Modelering MA1487 HT24
*Olga Egorova, oleg22*

## Introduction

I projektet förväntas vi att plocka data från en open API och berbeta de med statistiska metoder.

## Uppgift 1. Databeskrivning

Jag vädle att plocka data från [SMHI Open Data API Docs - Meteorological Observations](https://opendata.smhi.se/apidocs/metobs/index.html). Jag välde att plocka temperaturmätningar (parameter 1) och relativt luftfuktighet (parameter 6). Dessa mätningar pågar varje timme. Jag använder tre stationer: Halmstad flygplats, Uppsala Flygplats och Umeå Flygplats. Stations nämns som i SMHI Oen Data. Temperatur mäts i Celcie grad (°C) och Relativt luftfuktighet i procenter (%). Dataurval presenterades i [Tabel 1a](### Tabel 1a. TEMPERATUR per timme under sista tre dagar från tre stationer:) och [Tabel 1b](### Tabel 1b. LUFTFUKTIGHET per timme från tre stationer).
Koden till funktioner för att hämta data finns i [GitHub](https://github.com/OlganeOlga/MathMod/tree/master/get_dynam_data).

Alla tabeller och figurer skapas med filen [ALL_CODE.py](ALL_CODE.py)
### Tabel 1a. TEMPERATUR per timme under sista tre dagar från tre stationer:
(exampel)
|                     |   Halmstad flygplats(°C) |   Uppsala Flygplats(°C) |   Umeå Flygplats(°C) |
|:--------------------|-------------------------:|------------------------:|---------------------:|
| 2024-12-15 17:00:00 |                      7.8 |                    -2.3 |                 -6.8 |
| 2024-12-15 18:00:00 |                      8.1 |                    -1.8 |                 -4.4 |
| 2024-12-15 19:00:00 |                      8.2 |                    -1.1 |                 -3.1 |
| 2024-12-15 20:00:00 |                      8.4 |                     0.4 |                 -1.3 |
| 2024-12-15 21:00:00 |                      8.2 |                     1.2 |                 -2.3 |
.......
| 2024-12-18 12:00:00 |                      6.1 |                     0.5 |                 -7.9 |
| 2024-12-18 13:00:00 |                      6   |                     1.6 |                 -6.8 |
| 2024-12-18 14:00:00 |                      6.5 |                     2.3 |                 -4.1 |
| 2024-12-18 15:00:00 |                      7   |                     2.7 |                 -3.4 |
| 2024-12-18 16:00:00 |                      7.4 |                     3.4 |                 -3.1 |
 
### Tabel 1b. LUFTFUKTIGHET per timme från tre stationer
(exampel)
|                     |   Halmstad flygplats(%) |   Uppsala Flygplats(%) |   Umeå Flygplats(%) |
|:--------------------|------------------------:|-----------------------:|--------------------:|
| 2024-12-15 17:00:00 |                      98 |                     99 |                  90 |
| 2024-12-15 18:00:00 |                      95 |                    100 |                  92 |
| 2024-12-15 19:00:00 |                      94 |                    100 |                  93 |
| 2024-12-15 20:00:00 |                      94 |                    100 |                  96 |
| 2024-12-15 21:00:00 |                      93 |                    100 |                  95 |
.........
| 2024-12-18 11:00:00 |                      96 |                     95 |                  90 |
| 2024-12-18 12:00:00 |                      96 |                    100 |                  92 |
| 2024-12-18 13:00:00 |                      98 |                    100 |                  93 |
| 2024-12-18 14:00:00 |                      97 |                    100 |                  95 |
| 2024-12-18 15:00:00 |                      96 |                    100 |                  95 |
| 2024-12-18 16:00:00 |                      96 |                    100 |                  96 |

Jag använder pivottabel:
station_name              Halmstad flygplats            Umeå Flygplats            Uppsala Flygplats
parameter                      LUFTFUKTIGHET TEMPERATUR  LUFTFUKTIGHET TEMPERATUR     LUFTFUKTIGHET TEMPERATUR
time
2024-12-15 18:00:00+01:00               98.0        7.8           90.0       -6.8              99.0       -2.3
2024-12-15 19:00:00+01:00               95.0        8.1           92.0       -4.4             100.0       -1.8
2024-12-15 20:00:00+01:00               94.0        8.2           93.0       -3.1             100.0       -1.1
2024-12-15 21:00:00+01:00               94.0        8.4           96.0       -1.3             100.0        0.4
2024-12-15 22:00:00+01:00               93.0        8.2           95.0       -2.3             100.0        1.2
...                                      ...        ...            ...        ...               ...        ...
2024-12-18 13:00:00+01:00               96.0        6.1           92.0       -7.9             100.0        0.5
2024-12-18 14:00:00+01:00               98.0        6.0           93.0       -6.8             100.0        1.6
2024-12-18 15:00:00+01:00               97.0        6.5           95.0       -4.1             100.0        2.3
2024-12-18 16:00:00+01:00               96.0        7.0           95.0       -3.4             100.0        2.7
2024-12-18 17:00:00+01:00               96.0        7.4           96.0       -3.1             100.0        3.4


Jag tittar om det finns missade data för [temperatur](### Tabel 2a.) och för [relativt luftfuktighet](### Tabel 2b.)

### Tabel 2a. [Missade data för TEMPERATUR](statistics/TEMPERATUR_mis_summ.md)           
|                    |   0 |                
|:-------------------|----:|                
| Halmstad flygplats |   0 |                
| Umeå Flygplats     |   0 |               
| Uppsala Flygplats  |   0 |               

### Tabel 2b. [Missade data för RELATIVT LUFTFUKTIGHET](statistics/LUFTFUKTIGHET_mis_summ.md) 
|                    |   0 |
|:-------------------|----:|
| Halmstad flygplats |   0 |
| Umeå Flygplats     |   0 |
| Uppsala Flygplats  |   0 |

### Tabel 3c. [Missade data för alla parameter: ](statistics/ALLA_mis_summ.md)
|        station och parameter            |N missad|
|:----------------------------------------|-------:|
| ('Halmstad flygplats', 'LUFTFUKTIGHET') |   0    |
| ('Halmstad flygplats', 'TEMPERATUR')    |   0    |
| ('Umeå Flygplats', 'LUFTFUKTIGHET')     |   0    |
| ('Umeå Flygplats', 'TEMPERATUR')        |   0    |
| ('Uppsala Flygplats', 'LUFTFUKTIGHET')  |   0    |
| ('Uppsala Flygplats', 'TEMPERATUR')     |   0    |

Det verkar att inga tidspunkter var missad under dessa tre dagar.

Jag vill teasta om datamängd är normalfördelad. För detta skull använder jag Shapiro-Wilk test för normalitets sprigning.

### Tabel 3. [Beskrivande statistik for parameters](statistics/describe_stat_all.md)
station_name:          Halmstad flygplats            Umeå Flygplats            Uppsala Flygplats
parameter         LUFTFUKTIGHET TEMPERATUR  LUFTFUKTIGHET TEMPERATUR     LUFTFUKTIGHET TEMPERATUR
count                     72.00      72.00          72.00      72.00             72.00      72.00
mean                      91.47       6.91          88.38     -10.61             78.01       1.27
std                        5.98       0.93           4.10       5.68             14.14       2.48
min                       75.00       4.40          81.00     -20.40             57.00      -4.70
25%                       90.00       6.38          85.00     -15.82             64.00       0.18
50%                       93.00       7.00          88.00     -10.05             77.50       1.90
75%                       96.00       7.43          91.25      -5.38             87.25       2.72
max                       99.00       8.90          96.00      -1.30            100.00       6.60


Medelvärde i stationer Halmstad Flugplats och Upsala Flugplats är närmare medianen, som säger att de ssa data 
närmare normafördelning än data från Umeå Flugplats

![Ladogrammar för TEMPERATUR](img/box_plot/TEMPERATUR_combined_box_plots.png)

![Temperatur frekvenser](img/frekvenser/TEMPERATUR_combined.png)

*Med dessa plottar och Shapiro-Wilk test testar jag nulhypotes: att data är noirmalfördelad.*
Både plottar och Shapiro-Wilk test för normality tillåtar förkasta nulhypotes om att temperatur spridning är normal fördelad. Sannolikheten att nulhypotes stämmer är 3.78% för Halmstad flygplats, som är mindre än 5% och därmed är sannolikhet för typ II fel är ganska liten.
För andra två platser respectivt sannolikhheten för att nulhypotes stämmer är 0.29% och 0.02% och därmed är möjlighet för att felförkasta nulhypotes (fel typ II) är ännu mindre.
### Q_Q plottar
Det finns ett annat sät att visualisera avvikelse från normalfördelning, n-mligen [kvantil_kvantil plot](https://pubmed.ncbi.nlm.nih.gov/5661047/). Varje axel visar fördelningen av en dataset. I detta fall jämför jag dataset från olika stationer mot den teoretiska normalfördelningen. På X-axeln visas normafördelnings kvantiler, på Y-axeln visas kvantiler från respektiv datamängd (Tabel 3[a](### Tabel 3a)[b][### Tabel 3b])
### Fig 4a
![Kvanti_kventil ploter för TEMPERATUR](img/q_q_plot/TEMPERATUR_combined_qq_plots.png)

### Tabel 3b. [Beskrivande statistik RELATIVT LUFTFUKTIGHET](statistics/LUFTFUKTIGHET_describe_stat.md)
Om jag gör samma test för relativt lurftfuktighet visas det att luftfuktighet i Umeå Flugplats kan vara normalfördelad eftersom p_värde är 6.95% och större än 5%, dvs nulhypotes om att data är normalfördelade kan inta förkastas. Det är ppga stor sannoliket för fel typ II.

![Ladogrammar för relativt LUFTFUKTIGHET](img/box_plot/LUFTFUKTIGHET_combined_box_plots.png)

![Luftfuktighet frekvenser](img/frekvenser/LUFTFUKTIGHET_combined.png)

### Tabel 4b
![Kvanti_kventil ploter för RELATIVT LUFTFUKTIGHET](img/q_q_plot/LUFTFUKTIGHET_combined_qq_plots.png)

Dess plottar visa samma: aärmast till normalfördelningen är data från station Halmstad flygplats, för både temperatur och relativt lyftfuktighet.

# Uppgift 4: Linjär regression
jag ser hur korrelerar olika variabler med varandra
![Korrelation matrix](img/correlations/all_correlations.png)

Matrix visar att den bästa correlation är mellan temperatur och relativt luftfuktighet i Umeå.
Därför välde jag att använda dessa variabler för liniar regression

*Utför en linjärregression av minst en av variablerna och ett tillhörande 95% konfidensintervall. 
Rapportera variablerna 𝑎  och 𝑏  i sambandet 𝑦 = 𝑎 + 𝑏 ∙ 𝑥  samt punktskattningens 
konfidensintervall av dessa. Visualisera detta i en graf med den linjära modellen, konfidensintervallet 
och originaldata i samma figur.*

Jag gör liniar regression för relativt luft fuktighet i Umea Fluglats. Jag väljer det datamängd eftersom fördelningen i detta grupp data är normal med största sannolikhet.
