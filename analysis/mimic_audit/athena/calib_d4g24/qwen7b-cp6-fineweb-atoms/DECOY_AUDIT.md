# S2b decoy audit table (signed package)

run `qwen7b-cp6-fineweb-atoms`  claims 75  candidates 1335 (measured ok 1335)

signed: p_band=90 eps_pr=5.2251 fve_tol=0.0221 tau_val=144.051 kappa_0=0.8 (provisional) power=per-family Q95

coverage: resolved(>=5) 59/75, unresolved 16/75, powered(family) 60/75, powered AND resolved 50/75, band-disabled 14, capped 16

| idx.ci | type | claim | gen | text | pool | eps_ov | surv | power | supp_b | note |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.2 | detail | "So what's going on WITH STEPHANIE'S pregnancy on GH and STE | 24 | 22 | 22 | 0.1 | 10 | pass +48.2 | 141.19 |  |
| 0.4 | theme | implying a second show's title | 24 | 23 | 23 | 0.1 | 5 | fail -1.6 | 24.83 |  |
| 0.6 | theme | is a truncated word mid-sentence | 24 | 22 | 10 | 0.3 | 9 | fail -2.9 | 0.58 |  |
| 112.10 | theme | expecting a noun phrase like | 24 | 23 | 7 | 10 | 6 | pass +12.5 | 0.60 |  |
| 112.11 | detail | "shell company" | 24 | 23 | 23 | 1 | 5 | pass +3.7 | 30.77 |  |
| 112.4 | theme | suggesting a concluding clause about its merged or troubled | 24 | 14 | 9 | 1 | 8 | pass +27.9 | 86.40 |  |
| 128.1 | theme | implying a pattern of song titles or related content | 24 | 4 | 2 | band-off | 2 | pass +30.3 | 57.76 | UNRESOLVED (<5 decoys) |
| 128.5 | theme | closes a second example in a parallel list | 24 | 9 | 4 | band-off | 4 | pass +20.9 | 46.65 | UNRESOLVED (<5 decoys) |
| 128.6 | detail | "Tuesday April 15, 2015: 'The Weeknd' and 'Unofficial Music | 24 | 23 | 4 | band-off | 4 | pass +76.5 | 143.92 | UNRESOLVED (<5 decoys) |
| 144.12 | detail | "will automatically receive both items and the first one." | 24 | 20 | 20 | band-off | 20 | pass +29.7 | 78.36 |  |
| 144.4 | theme | implies a promised result or instruction about the cart beh | 24 | 20 | 19 | 2 | 5 | pass +43.3 | 72.22 |  |
| 144.5 | theme | likely completing the savings or item selection logic | 24 | 18 | 18 | 0.75 | 9 | pass +11.4 | 65.22 |  |
| 160.6 | theme | describing a technique to ensure the egg whites are beaten | 24 | 22 | 22 | 2 | 10 | pass +22.4 | 89.42 |  |
| 160.7 | detail | "looks" | 24 | 0 | 0 | - | 0 | n/a | 7.29 | UNRESOLVED (<5 decoys) |
| 160.8 | theme | ends an incomplete clause | 24 | 24 | 3 | band-off | 3 | fail -1.8 | 0.00 | UNRESOLVED (<5 decoys) |
| 16.11 | theme | is mid-item in a parenthetical list of monthly economic ind | 24 | 21 | 21 | 0.5 | 8 | pass +24.9 | 68.48 |  |
| 16.5 | theme | with standard economic indicators like | 24 | 14 | 14 | 0.5 | 7 | pass +7.7 | 62.66 |  |
| 16.7 | detail | "CPI" | 24 | 21 | 21 | 10 | 11 | pass +1.3 | 12.76 |  |
| 176.10 | theme | is an article mid-list | 24 | 0 | 0 | - | 0 | n/a | 9.65 | UNRESOLVED (<5 decoys) |
| 176.15 | detail | "complete curriculum package that includes worship material | 24 | 24 | 18 | 1 | 6 | pass +27.1 | 85.71 |  |
| 176.6 | theme | implies a list of two additional products or resources, | 24 | 13 | 10 | 0.75 | 8 | pass +14.4 | 48.03 |  |
| 192.10 | detail | " , 2008)" | 24 | 10 | 10 | 0.1 | 6 | pass +5.1 | 45.82 |  |
| 192.4 | theme | mirrors earlier incomplete lists of prior Olympic appearanc | 24 | 1 | 1 | band-off | 1 | pass +71.3 | 160.96 | UNRESOLVED (<5 decoys) |
| 192.5 | theme | Final token | 24 | 19 | 0 | - | 0 | n/a | 0.00 | UNRESOLVED (<5 decoys) |
| 208.3 | theme | implies a book or podcast title follows | 24 | 15 | 8 | 1.5 | 6 | pass +13.5 | 44.17 |  |
| 208.8 | theme | strongly expecting continuation like | 24 | 22 | 17 | 1 | 5 | fail -8.9 | 0.07 |  |
| 208.9 | detail | "The audio" | 24 | 23 | 23 | 0.5 | 8 | pass +4.9 | 26.02 |  |
| 224.1 | theme | using anecdotes to illustrate gear and attitude | 24 | 6 | 6 | band-off | 6 | fail -6.9 | 28.61 |  |
| 224.5 | theme | continuing the quoted example clause describing the initial | 24 | 16 | 9 | 1 | 7 | pass +29.3 | 140.82 |  |
| 224.7 | detail | "hold their own pitch individually" | 24 | 19 | 19 | 0.5 | 5 | pass +15.4 | 45.02 |  |
| 240.1 | theme | establishing scientific context around Dow AgroSciences's in | 24 | 24 | 24 | 0.15 | 6 | pass +5.8 | 93.92 |  |
| 240.4 | detail | Final token "and" is mid-list ("citrus, apples and"), part o | 24 | 22 | 6 | 0.3 | 6 | pass +45.3 | 186.85 |  |
| 240.5 | theme | strongly expecting a specific fruit or crop name like "other | 24 | 21 | 12 | 1.5 | 12 | pass +120.0 | 183.45 |  |
| 256.11 | detail | "Are you a student" | 24 | 24 | 24 | 0.75 | 11 | pass +53.7 | 62.24 |  |
| 256.5 | theme | The phrase "Are you a student" signals a common website UI p | 24 | 21 | 21 | 0.1 | 20 | pass +30.5 | 137.89 |  |
| 256.6 | theme | likely introducing a reader survey or eligibility context, | 24 | 13 | 13 | 1.5 | 5 | pass +17.4 | 63.67 |  |
| 272.0 | theme | Blog-style article tone with girl-focused fashion article fo | 24 | 7 | 7 | 0.75 | 5 | pass +26.9 | 69.76 |  |
| 272.10 | theme | strongly expecting | 24 | 24 | 20 | 0.1 | 6 | fail -10.8 | 1.11 |  |
| 272.11 | detail | "changing" | 24 | 23 | 23 | 1 | 9 | pass +11.1 | 20.97 |  |
| 288.0 | theme | Women's Magazine tone with witty headlines and gossip-y repo | 24 | 24 | 24 | 10 | 7 | pass +13.7 | 59.40 |  |
| 288.7 | detail | "this seemingly seems to" | 24 | 11 | 11 | 2 | 6 | pass +60.2 | 66.50 |  |
| 288.8 | theme | part of a concluding summary phrase | 24 | 24 | 24 | 0.3 | 7 | fail -1.2 | 16.42 |  |
| 304.0 | theme | Baseball/sports blog format with sports journalism article s | 24 | 14 | 14 | 0.3 | 5 | pass +5.1 | 46.03 |  |
| 304.10 | detail | ("335/.335/. ") | 24 | 24 | 24 | 2 | 7 | pass +5.5 | 61.07 |  |
| 304.13 | theme | expecting continuation like | 24 | 24 | 6 | 2 | 6 | fail -7.7 | -0.25 |  |
| 320.2 | theme | building a historical timeline of Microsoft Silverlight's e | 24 | 4 | 4 | band-off | 4 | pass +13.2 | 76.90 | UNRESOLVED (<5 decoys) |
| 320.4 | detail | "So WPF was never really a browser plug-in now its Silverlig | 24 | 20 | 2 | band-off | 2 | pass +68.8 | 113.91 | UNRESOLVED (<5 decoys) |
| 320.6 | theme | continuing a list of Microsoft's past technology approaches | 24 | 24 | 24 | 0.75 | 6 | pass +15.5 | 71.86 |  |
| 32.15 | theme | — a celebrity gossip item | 24 | 15 | 15 | 3 | 6 | pass +7.1 | 30.48 |  |
| 32.3 | theme | about Lindsay Lohan's DUI arrest. | 24 | 24 | 24 | 0.2 | 8 | pass +2.6 | 46.12 |  |
| 32.4 | detail | "Britney Spears' DUI video: Lindsay Lohan's" | 24 | 23 | 23 | 0.5 | 5 | pass +2.5 | 98.26 |  |
| 336.3 | theme | suggests a concluding clause about counting attendance or i | 24 | 21 | 21 | 0.5 | 5 | pass +25.9 | 41.74 |  |
| 336.5 | theme | implying a personal note about the poll's purpose or timing | 24 | 14 | 1 | band-off | 1 | pass +10.4 | 34.72 | UNRESOLVED (<5 decoys) |
| 336.9 | detail | "I'm curious to... I'm curious to" | 24 | 15 | 13 | 0.5 | 9 | pass +45.3 | 79.81 |  |
| 352.2 | theme | recounting a space shuttle crew's tense moment as a storm a | 24 | 24 | 19 | 0.2 | 5 | pass +20.0 | 83.59 |  |
| 352.5 | theme | implying a cause or action about medical readiness. | 24 | 21 | 21 | 0.3 | 5 | pass +15.1 | 53.78 |  |
| 352.8 | detail | "one with an assistant and a second person, the first two ru | 24 | 21 | 19 | 0.3 | 7 | pass +73.1 | 194.67 |  |
| 368.15 | theme | completing the informal invitation structure. | 24 | 24 | 24 | 0.2 | 5 | pass +11.7 | 35.37 |  |
| 368.4 | theme | implies a closing invitation or greeting | 24 | 21 | 15 | 1 | 6 | pass +26.2 | 39.03 |  |
| 368.6 | detail | "anytime." Final token | 24 | 23 | 4 | band-off | 4 | pass +12.6 | 23.79 | UNRESOLVED (<5 decoys) |
| 384.4 | theme | implying a second charge was withdrawn or resolved as part o | 24 | 19 | 18 | 5 | 9 | pass +23.5 | 84.35 |  |
| 384.7 | theme | ends an incomplete phrase | 24 | 0 | 0 | - | 0 | n/a | 0.65 | UNRESOLVED (<5 decoys) |
| 384.8 | detail | ("has not been part of the plea deal as part of") | 24 | 4 | 3 | band-off | 3 | pass +57.3 | 105.46 | UNRESOLVED (<5 decoys) |
| 48.12 | detail | "could certainly be added to the list." | 24 | 23 | 23 | 1.5 | 9 | pass +8.4 | 45.84 |  |
| 48.3 | theme | mirrors a second list item, | 24 | 24 | 24 | 0.5 | 5 | pass +0.1 | 33.04 |  |
| 48.7 | theme | is a conjunction mid-clause | 24 | 24 | 23 | 0.2 | 6 | fail -1.8 | 1.82 |  |
| 64.0 | theme | Technical/Forum post response with a programmer trying to fi | 24 | 24 | 24 | 0.1 | 7 | pass +28.2 | 187.41 |  |
| 64.2 | theme | The sentence structure | 24 | 20 | 1 | band-off | 1 | fail -7.5 | -0.11 | UNRESOLVED (<5 decoys) |
| 64.9 | detail | "I would stay away," | 24 | 1 | 1 | band-off | 1 | pass +15.0 | 70.04 | UNRESOLVED (<5 decoys) |
| 80.1 | theme | with practical examples around service quality and customer | 24 | 17 | 11 | 0.5 | 7 | pass +13.1 | 71.95 |  |
| 80.2 | theme | using a structured list format with numbered points. | 24 | 24 | 24 | 1 | 6 | pass +8.1 | 42.26 |  |
| 80.5 | detail | "factor. " | 24 | 19 | 19 | 0.3 | 6 | pass +0.3 | 11.25 |  |
| 96.1 | theme | establishing personal anecdotes about | 24 | 6 | 6 | 2 | 5 | fail -3.7 | 32.22 |  |
| 96.12 | detail | "am forced to admit the plant is absent" | 24 | 23 | 23 | 0.3 | 6 | pass +19.5 | 59.14 |  |
| 96.6 | theme | with a humorous aside about inventory. | 24 | 22 | 22 | 0.1 | 5 | pass +11.4 | 35.09 |  |
