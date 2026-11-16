"""Phase 4: the simulation dashboard (manual Section 11, DECISIONS D-32 to D-34, D-39 to D-41).

The dashboard is a replay of logs written by the evaluation harness. Nothing in this package runs
a simulator rule on its own: ``store`` reads files, ``derive`` rearranges log rows, ``figures``
draws them, and ``custom`` hands a custom shock to the harness.
"""
