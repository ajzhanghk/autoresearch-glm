# ODLS(10)-literature common-transversal (ct) mining

ct = number of transversals of L whose cells also carry 10 distinct values in B
(i.e., common transversals of the orthogonal pair (L,B)).  A pair extends to a
third mutually orthogonal square iff it has 10 disjoint common transversals;
the published record for an orthogonal pair of order 10 is ct = 7.

Pairs scanned: 10481 verified orthogonal pairs of 10x10 Latin squares.
Global ct histogram: {0: 5899, 1: 3363, 2: 961, 3: 220, 4: 32, 5: 6}
Max ct found: 5

## Sources scanned

- turn-square(brown) from olegzaikin/latinsquares DMD-2016 quarter: 4000 pairs, hist {0: 2229, 1: 1300, 2: 371, 3: 88, 4: 10, 5: 2}
- turn-square(dmd) from olegzaikin/latinsquares DMD-2016 quarter: 4000 pairs, hist {0: 2251, 1: 1283, 2: 368, 3: 87, 4: 9, 5: 2}
- random turn-square family (LS(5) pair-structure quarters, Brown-et-al construction): 2266 pairs, hist {0: 1246, 1: 761, 2: 200, 3: 44, 4: 13, 5: 2}
- gridcoin-community/odlk-progs ortogon/mout.txt (explicit DLK+mate pairs): 21 pairs, hist {2: 20, 3: 1}
- odlk mar_abk DB, blind sample, computed mates: 33 pairs, hist {0: 29, 1: 3, 2: 1}
- odlk mar_abk DB, top-300 by transversal count, computed mates: 161 pairs, hist {0: 144, 1: 16, 2: 1}

Notes:
- gridcoin-community/odlk-progs is tooling+data of the ODLK BOINC project
  (Belyshev canonical forms); psevdoass/mar_abk_4.3.7.2.1.9.5.6.txt contains
  304,574 canonical DLS(10) that possess orthogonal mates.  Mates were
  recomputed here as decompositions of L into 10 disjoint transversals
  (every decomposition is an orthogonal mate up to relabelling; ct is
  relabelling-invariant).  Decomposition enumeration was capped per square.
- turn squares (Brown-et-al style, quarter = LS(5) pair-structure) are the
  transversal-rich class (4224 transversals vs ~860 median for ODLK DB
  squares) and produced every ct >= 4 pair found.
- olegzaikin/latinsquares embeds the DMD-2016 quarter used by Zaikin's
  enumerate_brown_dls tool.

## Saved pairs (ct >= 4)

| file | ct | provenance |
|------|----|------------|
| pair_1.json | 5 | turn-square(brown) from olegzaikin/latinsquares DMD-2016 quarter |
| pair_2.json | 5 | turn-square(brown) from olegzaikin/latinsquares DMD-2016 quarter |
| pair_3.json | 5 | turn-square(dmd) from olegzaikin/latinsquares DMD-2016 quarter |
| pair_4.json | 5 | turn-square(dmd) from olegzaikin/latinsquares DMD-2016 quarter |
| pair_5.json | 5 | random turn-square family (LS(5) pair-structure quarters, Brown-et-al construction) quarter=[[8, 3, 4, 2, 0], [5, 1, 7, 9, 6], [2, 5, 0, 3, 1], [3, 0, 8, 4, 2], [0, 7, 3, 8, 5]], style=dmd |
| pair_6.json | 5 | random turn-square family (LS(5) pair-structure quarters, Brown-et-al construction) quarter=[[9, 5, 6, 8, 2], [3, 9, 2, 5, 1], [4, 1, 0, 7, 6], [8, 7, 5, 6, 0], [7, 6, 8, 9, 4]], style=brown |
| pair_7.json | 4 | turn-square(brown) from olegzaikin/latinsquares DMD-2016 quarter |
| pair_8.json | 4 | turn-square(brown) from olegzaikin/latinsquares DMD-2016 quarter |
| pair_9.json | 4 | turn-square(brown) from olegzaikin/latinsquares DMD-2016 quarter |
| pair_10.json | 4 | turn-square(brown) from olegzaikin/latinsquares DMD-2016 quarter |
| pair_11.json | 4 | turn-square(brown) from olegzaikin/latinsquares DMD-2016 quarter |
| pair_12.json | 4 | turn-square(brown) from olegzaikin/latinsquares DMD-2016 quarter |
| pair_13.json | 4 | turn-square(brown) from olegzaikin/latinsquares DMD-2016 quarter |
| pair_14.json | 4 | turn-square(brown) from olegzaikin/latinsquares DMD-2016 quarter |
| pair_15.json | 4 | turn-square(brown) from olegzaikin/latinsquares DMD-2016 quarter |
| pair_16.json | 4 | turn-square(brown) from olegzaikin/latinsquares DMD-2016 quarter |
| pair_17.json | 4 | turn-square(dmd) from olegzaikin/latinsquares DMD-2016 quarter |
| pair_18.json | 4 | turn-square(dmd) from olegzaikin/latinsquares DMD-2016 quarter |
| pair_19.json | 4 | turn-square(dmd) from olegzaikin/latinsquares DMD-2016 quarter |
| pair_20.json | 4 | turn-square(dmd) from olegzaikin/latinsquares DMD-2016 quarter |
| pair_21.json | 4 | turn-square(dmd) from olegzaikin/latinsquares DMD-2016 quarter |
| pair_22.json | 4 | turn-square(dmd) from olegzaikin/latinsquares DMD-2016 quarter |
| pair_23.json | 4 | turn-square(dmd) from olegzaikin/latinsquares DMD-2016 quarter |
| pair_24.json | 4 | turn-square(dmd) from olegzaikin/latinsquares DMD-2016 quarter |
| pair_25.json | 4 | turn-square(dmd) from olegzaikin/latinsquares DMD-2016 quarter |
| pair_26.json | 4 | random turn-square family (LS(5) pair-structure quarters, Brown-et-al construction) quarter=[[5, 1, 9, 2, 3], [6, 5, 1, 9, 2], [2, 3, 4, 8, 9], [8, 0, 2, 6, 4], [0, 7, 6, 5, 8]], style=brown |
| pair_27.json | 4 | random turn-square family (LS(5) pair-structure quarters, Brown-et-al construction) quarter=[[3, 5, 1, 0, 7], [2, 6, 9, 5, 1], [5, 0, 2, 1, 3], [8, 7, 5, 6, 0], [0, 8, 6, 7, 4]], style=dmd |
| pair_28.json | 4 | random turn-square family (LS(5) pair-structure quarters, Brown-et-al construction) quarter=[[8, 3, 4, 2, 0], [5, 1, 7, 9, 6], [2, 5, 0, 3, 1], [3, 0, 8, 4, 2], [0, 7, 3, 8, 5]], style=dmd |
| pair_29.json | 4 | random turn-square family (LS(5) pair-structure quarters, Brown-et-al construction) quarter=[[4, 8, 7, 9, 3], [1, 3, 5, 2, 9], [6, 9, 1, 4, 2], [2, 4, 0, 6, 8], [0, 7, 3, 1, 4]], style=brown |
| pair_30.json | 4 | random turn-square family (LS(5) pair-structure quarters, Brown-et-al construction) quarter=[[3, 8, 9, 2, 5], [7, 9, 5, 1, 3], [8, 4, 6, 0, 7], [9, 3, 7, 5, 1], [4, 7, 8, 3, 0]], style=dmd |
| pair_31.json | 4 | random turn-square family (LS(5) pair-structure quarters, Brown-et-al construction) quarter=[[4, 3, 9, 7, 8], [6, 9, 7, 8, 5], [7, 8, 5, 6, 9], [9, 7, 1, 4, 6], [8, 5, 3, 0, 7]], style=brown |
| pair_32.json | 4 | random turn-square family (LS(5) pair-structure quarters, Brown-et-al construction) quarter=[[4, 3, 9, 7, 8], [6, 9, 7, 8, 5], [7, 8, 5, 6, 9], [9, 7, 1, 4, 6], [8, 5, 3, 0, 7]], style=brown |
| pair_33.json | 4 | random turn-square family (LS(5) pair-structure quarters, Brown-et-al construction) quarter=[[5, 0, 3, 1, 7], [1, 5, 9, 7, 3], [0, 6, 7, 4, 8], [6, 7, 8, 9, 5], [2, 8, 4, 6, 0]], style=brown |
| pair_34.json | 4 | random turn-square family (LS(5) pair-structure quarters, Brown-et-al construction) quarter=[[5, 0, 3, 1, 7], [1, 5, 9, 7, 3], [0, 6, 7, 4, 8], [6, 7, 8, 9, 5], [2, 8, 4, 6, 0]], style=brown |
| pair_35.json | 4 | random turn-square family (LS(5) pair-structure quarters, Brown-et-al construction) quarter=[[2, 5, 6, 8, 9], [6, 9, 8, 5, 2], [9, 1, 2, 6, 5], [4, 6, 9, 7, 1], [8, 7, 5, 9, 3]], style=dmd |
| pair_36.json | 4 | random turn-square family (LS(5) pair-structure quarters, Brown-et-al construction) quarter=[[2, 8, 0, 6, 5], [3, 5, 2, 1, 9], [5, 2, 1, 0, 3], [9, 3, 4, 7, 8], [8, 9, 3, 5, 2]], style=brown |
| pair_37.json | 4 | random turn-square family (LS(5) pair-structure quarters, Brown-et-al construction) quarter=[[7, 4, 1, 9, 6], [0, 7, 6, 8, 5], [1, 9, 4, 3, 2], [4, 6, 0, 2, 8], [3, 1, 2, 4, 9]], style=brown |
| pair_38.json | 4 | random turn-square family (LS(5) pair-structure quarters, Brown-et-al construction) quarter=[[7, 9, 3, 1, 4], [3, 7, 4, 9, 1], [5, 6, 8, 2, 0], [9, 8, 2, 4, 6], [8, 4, 0, 3, 7]], style=brown |
