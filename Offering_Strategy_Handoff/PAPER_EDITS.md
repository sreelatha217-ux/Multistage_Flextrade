# Paper edits: remove synthetic-data material

## Figure replacement map (run real_data_figures.py)
| Old figure (synthetic day)                | New figure (reference tree 2026-10-11, real DK1) |
|-------------------------------------------|---------------------------------------------------|
| Fig. 2  Executive dashboard               | fig_real_dashboard                                |
| Fig. 3  Stage-1 Gantt + MT + DA curves    | fig_real_stage1  (panel c also replaces Fig. 4)   |
| Fig. 4  DA offer curves                   | delete (now panel c of fig_real_stage1)           |
| Fig. 5  Reserve by asset                  | fig_real_reserve                                  |
| Fig. 6  BESS dispatch, Fig. 7 MT          | fig_real_bess_mt (one figure, two panels)         |
| Fig. 8  Stage-2 recourse                  | fig_real_stage2                                   |
| Fig. 9  Stage-3, Fig. 10 profit dist.     | fig_real_stage3 (one figure, four panels)         |
| Fig. 11, Fig. 12 rescheduling (synthetic) | delete; real version is panel (c) of dashboard + Table XIII |
Fig. 1 (seasonal) is already real data: keep.

## Text to delete or rewrite
1. Contribution 3 (Sec. II-D): delete the sentence "...the total saving is less than half of what the earlier synthetic assumption (reserve price 0.12x Day-Ahead) predicted for the same tree (2.9% against 6.7%)."
2. Sec. VI-A: delete "...instead of the 12% assumed in the first version of this study."
   Keep: "...C6.79/MW/h, or 5.2% of the Day-Ahead price."
3. Sec. VI-H "Reserve-price source and sensitivity": delete "The first version of this study set the reserve capacity price to 0.12 times the Day-Ahead buy price." Start with: "Real mFRR capacity prices on the reference tree average only 5.2% of the Day-Ahead price..."
   Table XI: delete the row "Synthetic 0.12xDA, price-quantity curve", and rewrite the sentence that quotes 6.69% -> 3.60% -> 2.89% to start from the real price-quantity curve (3.60%) -> Stage-1 quantity (2.89%).
4. Sec. VI-I: replace "This test uses synthetic Intraday prices" with "Intraday prices are modelled as in Sec. VI-A."
5. Limitation 8: same replacement ("...uses synthetic Intraday prices" -> "...uses modelled Intraday prices").
6. Delete Sec. VI-J "Illustrative Synthetic Day (First Version of the Study)" completely, plus the old Fig. 2-12 captions (the Fig. 2 caption is also pasted three times in the PDF).
7. Sec. V-C: rename "Benchmark Case Study Data" -> "Case-Study Plant and Asset Parameters" and add one sentence: "The plant (furnace durations, power draws, yields) is a representative parameter set, not measured data from a specific factory; only the market prices are real." A reviewer will ask.
8. Delete any remaining "first version", "earlier version", "synthetic benchmark" wording (search for: synthetic, first version, earlier, benchmark day).
