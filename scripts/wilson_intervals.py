"""
95% Wilson score intervals for the held-out recall and GMCF values
(Tables 6 and 7), written as LaTeX for Appendix C.4.

Counts are recovered from the published proportions and denominators and
checked to reproduce every printed value. Better still: call wilson() inside
the table harness on the raw counts.
"""
import math

Z = 1.959963984540054  # two-sided 95%


def wilson(k, n, z=Z):
    """Wilson score interval for k successes in n trials."""
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


def count(p, n, label):
    k = round(p * n)
    assert abs(k / n - p) < 0.0005 + 1e-9, f"{label}: {p} x {n} does not round-trip"
    return k


def ci(k, n):
    lo, hi = wilson(k, n)
    return f"[{lo:.2f}, {hi:.2f}]"


N_GEO = 151  # \bsTestInView

# Table 6: pre-registered geometry.
# label, R_PRE, GMCF_PRE, then (n_hit, n_clusters) for the arms, or for the
# descriptive rows the infix of the \bsDesc<infix>Test{Matched,Clusters,GmcfLo,
# GmcfHi} macros, which manuscript_numbers.py emits from holdout_pieranch.json
# using this file's wilson().
# Arm cluster counts are \bsArm*TestClusters in numbers.tex; n_hit is pinned by
# both the GMCF value and \bsArm*TestGeotagsPerCluster (found / n_hit).
T6 = [
    ("013", 0.583, 0.189, (69, 365)),
    ("014", 0.563, 0.223, (69, 309)),
    ("030 ($t=0.04$)", 0.887, 0.058, (106, 1818)),
    ("015", 0.815, 0.173, "Fifteen"),
    ("017", 0.695, 0.235, "Seventeen"),
    ("021", 0.702, 0.164, "TwentyOne"),
    ("028", 0.669, 0.315, "TwentyEight"),
    ("030 ($t=0.06$)", 0.762, 0.198, "ThirtyTSix"),
    ("031", 0.682, 0.235, "ThirtyOne"),
    ("042", 0.649, 0.185, "FortyTwo"),
    ("044", 0.689, 0.217, "FortyFour"),
    ("045", 0.689, 0.178, "FortyFive"),
]
GEOTAGS_PER_CLUSTER = {"013": 1.275, "014": 1.232, "030 ($t=0.04$)": 1.264}

# Table 7: review parameters.
# label, clusters, GMCF_PRE, R_PRE, GMCF_POST (pos, maj, neg)
T7 = [
    ("013", 132, 0.432, 0.490, (0.780, 0.742, 0.705)),
    ("014", 113, 0.522, 0.536, (0.867, 0.832, 0.832)),
    ("030 ($t=0.04$)", 445, 0.189, 0.868, (0.404, 0.339, 0.279)),
    ("015", 201, 0.373, 0.682, (0.726, 0.682, 0.632)),
    ("017", 225, 0.324, 0.715, (0.684, 0.627, 0.578)),
    ("021", 295, 0.247, 0.742, (0.539, 0.478, 0.417)),
    ("028", 382, 0.202, 0.801, (0.455, 0.393, 0.332)),
    ("030 ($t=0.06$)", 269, 0.279, 0.735, (0.617, 0.558, 0.509)),
    ("031", 215, 0.349, 0.728, (0.698, 0.647, 0.595)),
    ("042", 127, 0.409, 0.497, (0.819, 0.780, 0.748)),
    ("044", 294, 0.231, 0.742, (0.466, 0.384, 0.323)),
    ("045", 192, 0.396, 0.656, (0.750, 0.693, 0.635)),
]

rows6 = []
for i, (ck, r, g, counts) in enumerate(T6):
    k = count(r, N_GEO, f"T6 R {ck}")
    if isinstance(counts, tuple):
        h, n = counts
        assert round(h / n, 3) == g, f"T6 GMCF {ck}"
        assert round(k / h, 3) == GEOTAGS_PER_CLUSTER[ck], f"T6 geotags/cluster {ck}"
        gcells = f"{h}/{n} & {g:.3f} & {ci(h, n)}"
    else:
        m = rf"\bsDesc{counts}Test"
        gcells = rf"{m}Matched/{m}Clusters & {g:.3f} & [{m}GmcfLo, {m}GmcfHi]"
    rows6.append(f"        {ck} & {k}/{N_GEO} & {r:.3f} & {ci(k, N_GEO)} & {gcells} \\\\")
    if i == 2:
        rows6.append(r"        \midrule")

rows7 = []
for i, (ck, n, g_pre, r_pre, posts) in enumerate(T7):
    kr = count(r_pre, N_GEO, f"T7 R {ck}")
    kg = count(g_pre, n, f"T7 GMCF {ck}")
    kp = [count(g, n, f"T7 POST {ck}") for g in posts]
    cells = [ci(kr, N_GEO), ci(kg, n)] + [ci(k, n) for k in kp]
    rows7.append(f"        {ck} & {n} & " + " & ".join(cells) + r" \\")
    if i == 2:
        rows7.append(r"        \midrule")

tex = r"""\subsection{Sampling Uncertainty}
\label{app:intervals}

{\color{blue}Tables~\ref{tab:wilson_prereg} and~\ref{tab:wilson_review} give 95\% Wilson score intervals~\cite{Wilson1927} for the recall and \bsGmcfName{} values of Tables~\ref{tab:holdout} and~\ref{tab:adjudication}. 
Wilson intervals keep close to their nominal coverage at small sample sizes, where the familiar normal approximation does not~\cite{Brown2001}. 
Each value is a proportion of independent trials: for recall, the trials are the \bsTestInView{} in-view geotags, each found or missed; for \bsGmcfName{}, they are the detection clusters, each matched to a geotag (or, after review, confirmed) or not. 
The intervals describe sampling uncertainty only, that is, how much a value would vary had the flight contained a different set of burrows of the same kind. 
They do not capture the two systematic effects described in Appendix~\ref{app:metrics}: the survey-omission bias that makes \bsGmcfName{} a lower bound on precision, and the censoring that makes R$_{\mathrm{POST}}$ an upper bound on recall; for that reason no interval is given for R$_{\mathrm{POST}}$. 
Nor do they capture the choice of tie-break convention, whose effect is shown by reporting all three. 
Because burrows and detections cluster spatially, trials are not strictly independent, and the intervals should be read as approximate.}

\begin{table}[h]
    \centering
    \footnotesize
    \caption{\textcolor{blue}{Held-out values at the pre-registered geometry (Table~\ref{tab:holdout}) with 95\% Wilson intervals. Recall is over the \bsTestInView{} in-view geotags; \bsGmcfName{} is over the detection clusters, of which ``Matched'' contain at least one geotag. Operating points are those of Table~\ref{tab:holdout}. The top three rows are the pre-registered confirmatory arms.}}
    \label{tab:wilson_prereg}
    \setlength{\tabcolsep}{4pt}
    \begin{tabular}{lcccccc}
        \toprule
        \textbf{Ckpt} & \textbf{Found} & \textbf{R$_{\mathrm{PRE}}$} & \textbf{95\% CI} & \textbf{Matched} & \textbf{\bsGmcfName$_{\mathrm{PRE}}$} & \textbf{95\% CI} \\
        \midrule
""" + "\n".join(rows6) + r"""
        \bottomrule
    \end{tabular}
\end{table}

\begin{table}[h]
    \centering
    \footnotesize
    \caption{\textcolor{blue}{95\% Wilson intervals for the exploratory values of Table~\ref{tab:adjudication}, at the review parameters. $n$ is the number of detection clusters, the denominator of \bsGmcfName{}; recall is over the \bsTestInView{} in-view geotags. POST columns follow the tie-break conventions positive-wins / majority-wins / hard-negative-wins. R$_{\mathrm{POST}}$ is an upper bound and is given no interval.}}
    \label{tab:wilson_review}
    \setlength{\tabcolsep}{4pt}
    \begin{tabular}{lcccccc}
        \toprule
        & & & & \multicolumn{3}{c}{\textbf{\bsGmcfName$_{\mathrm{POST}}$}} \\
        \cmidrule(lr){5-7}
        \textbf{Ckpt} & $n$ & \textbf{R$_{\mathrm{PRE}}$} & \textbf{\bsGmcfName$_{\mathrm{PRE}}$} & \textbf{pos} & \textbf{maj} & \textbf{neg} \\
        \midrule
""" + "\n".join(rows7) + r"""
        \bottomrule
    \end{tabular}
\end{table}
"""

if __name__ == "__main__":
    open("appendix_wilson.tex", "w").write(tex)
    print("wrote appendix_wilson.tex")
