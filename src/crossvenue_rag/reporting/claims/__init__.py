"""What each document claims, as functions of Doc calls, and the values several documents share.

A claim function passes values from `reporting.compute` or from the pipeline function that
defines them (a term count, the McNemar power model), which `formatting` formats, to Doc calls
that compare them with the document.
"""
