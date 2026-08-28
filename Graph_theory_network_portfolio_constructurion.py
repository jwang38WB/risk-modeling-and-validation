"""
Graph Theory & Network-Based Portfolio Construction
===================================================================
Production module for:
- Correlation network construction (Mantegna distance)
- Minimum Spanning Tree (MST) analysis
- Network centrality measures
- Community detection (hierarchical + Louvain)
- Hierarchical Risk Parity (HRP) — de Prado 2016
- Dynamic network evolution and systemic risk monitoring

"""

from __future__ import annotations
import numpy as np
import pandas as pd
import networkx as nx
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
import logging, warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
class CorrelationNetwork:
    """
    Build and analyse correlation-based asset networks.
    Uses Mantegna (1999) distance metric: d(i,j) = sqrt(2*(1 - ρ))
    """

    def __init__(self, min_corr: float = 0.20):
        self.min_corr = min_corr
        self.G_: nx.Graph | None = None
        self.mst_: nx.Graph | None = None
        self.corr_matrix_: pd.DataFrame | None = None
        self.dist_matrix_: np.ndarray | None = None

    @staticmethod
    def mantegna_distance(rho: float) -> float:
        """Mantegna (1999) distance: d = sqrt(2*(1 - ρ))."""
        return float(np.sqrt(2.0 * (1.0 - rho)))

    def fit(self, returns: pd.DataFrame) -> "CorrelationNetwork":
        """Build the full correlation network from returns."""
        self.corr_matrix_ = returns.corr()
        corr = self.corr_matrix_.values
        n    = corr.shape[0]
        tickers = returns.columns.tolist()

        self.dist_matrix_ = np.sqrt(2 * (1 - np.clip(corr, -1, 1)))
        np.fill_diagonal(self.dist_matrix_, 0)

        G = nx.Graph()
        G.add_nodes_from(tickers)
        for i in range(n):
            for j in range(i+1, n):
                if corr[i, j] > self.min_corr:
                    d = self.mantegna_distance(corr[i, j])
                    G.add_edge(tickers[i], tickers[j],
                                weight=d, correlation=corr[i, j])
        self.G_ = G

        # Compute MST
        G_full = nx.Graph()
        for i in range(n):
            for j in range(i+1, n):
                G_full.add_edge(tickers[i], tickers[j],
                                 weight=self.dist_matrix_[i, j])
        self.mst_ = nx.minimum_spanning_tree(G_full, weight="weight")
        logger.info(f"Network: {n} nodes, {G.number_of_edges()} edges "
                     f"(min_corr>{self.min_corr}), MST: {self.mst_.number_of_edges()} edges")
        return self

    def centrality(self) -> pd.DataFrame:
        """Compute multiple centrality measures for the full network."""
        G = self.G_
        if G is None:
            raise RuntimeError("Call fit() first.")
        if not nx.is_connected(G):
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

        data = {
            "Degree":      dict(G.degree()),
            "Betweenness": nx.betweenness_centrality(G, weight="weight"),
            "Closeness":   nx.closeness_centrality(G, distance="weight"),
            "PageRank":    nx.pagerank(G, weight="weight"),
            "ClusterCoef": nx.clustering(G, weight="weight"),
        }
        try:
            data["Eigenvector"] = nx.eigenvector_centrality(G, weight="weight", max_iter=2000)
        except nx.PowerIterationFailedConvergence:
            data["Eigenvector"] = {n: np.nan for n in G.nodes()}

        return pd.DataFrame(data)

    def mst_centrality(self) -> pd.DataFrame:
        """Centrality measures on the MST (sparser, more interpretable)."""
        mst = self.mst_
        if mst is None:
            raise RuntimeError("Call fit() first.")
        return pd.DataFrame({
            "Degree":    dict(mst.degree()),
            "Betweenness": nx.betweenness_centrality(mst, weight="weight"),
        })

    def communities(self, n_clusters: int = 5,
                     method: str = "ward") -> pd.DataFrame:
        """
        Hierarchical community detection via linkage on Mantegna distance.

        Returns DataFrame with Asset → Community mapping.
        """
        if self.dist_matrix_ is None:
            raise RuntimeError("Call fit() first.")
        condensed = squareform(self.dist_matrix_, checks=False)
        Z = linkage(condensed, method=method)
        tickers = list(self.G_.nodes()) if self.G_ else []
        labels  = fcluster(Z, n_clusters, criterion="maxclust")
        df = pd.DataFrame({"Asset": tickers, "Community": labels})
        # Sort communities by size (largest first)
        sizes = df.groupby("Community").size().sort_values(ascending=False)
        rank_map = {c: i+1 for i, c in enumerate(sizes.index)}
        df["Community"] = df["Community"].map(rank_map)
        return df.sort_values("Community")

    def network_stats(self) -> dict:
        """Aggregate network topology statistics."""
        G = self.G_
        if G is None:
            raise RuntimeError("Call fit() first.")
        n = G.number_of_nodes()
        m = G.number_of_edges()
        density = nx.density(G)
        avg_corr = (self.corr_matrix_.values.sum() - n) / (n**2 - n)
        comps = nx.number_connected_components(G)
        if nx.is_connected(G):
            avg_path = nx.average_shortest_path_length(G, weight="weight")
        else:
            lcc = G.subgraph(max(nx.connected_components(G), key=len))
            avg_path = nx.average_shortest_path_length(lcc, weight="weight") if len(lcc) > 1 else np.nan
        return {"nodes": n, "edges": m, "density": density,
                "avg_correlation": avg_corr, "components": comps,
                "avg_shortest_path": avg_path}


# ─────────────────────────────────────────────────────────────────────────────
class HRPPortfolio:
    """
    Hierarchical Risk Parity (de Prado, 2016).

    Steps:
    1. Hierarchical clustering on Mantegna distance matrix
    2. Quasi-diagonalise the covariance matrix
    3. Recursive bisection to allocate weights
    """

    def __init__(self, linkage_method: str = "ward"):
        self.linkage_method = linkage_method
        self.weights_: pd.Series | None = None
        self.linkage_matrix_: np.ndarray | None = None
        self.sorted_idx_: list[int] | None = None

    def _get_cluster_var(self, cov: np.ndarray, idx: list[int]) -> float:
        """Inverse-volatility portfolio variance for a cluster."""
        sub_cov = cov[np.ix_(idx, idx)]
        inv_d   = 1.0 / (np.diag(sub_cov) + 1e-10)
        w_ivp   = inv_d / inv_d.sum()
        return float(w_ivp @ sub_cov @ w_ivp)

    def _recursive_bisect(self, cov: np.ndarray, sorted_idx: list[int],
                            weights: np.ndarray) -> None:
        """Recursively split the sorted asset list and allocate weights."""
        if len(sorted_idx) <= 1:
            return
        mid   = len(sorted_idx) // 2
        left  = sorted_idx[:mid]
        right = sorted_idx[mid:]
        v_l   = self._get_cluster_var(cov, left)
        v_r   = self._get_cluster_var(cov, right)
        alpha = 1.0 - v_l / (v_l + v_r)
        weights[left]  *= alpha
        weights[right] *= (1.0 - alpha)
        self._recursive_bisect(cov, left, weights)
        self._recursive_bisect(cov, right, weights)

    def fit(self, returns: pd.DataFrame) -> "HRPPortfolio":
        """Compute HRP weights from a returns DataFrame."""
        corr = returns.corr().values
        cov  = returns.cov().values
        n    = len(returns.columns)

        dist = np.sqrt(2 * (1 - np.clip(corr, -1, 1)))
        np.fill_diagonal(dist, 0)
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method=self.linkage_method)
        self.linkage_matrix_ = Z

        dn = dendrogram(Z, no_plot=True)
        sorted_idx = dn["leaves"]
        self.sorted_idx_ = sorted_idx

        weights = np.ones(n)
        self._recursive_bisect(cov, sorted_idx, weights)
        weights = weights / weights.sum()

        self.weights_ = pd.Series(weights, index=returns.columns, name="HRP")
        logger.info(f"HRP fitted: max_w={weights.max():.3f}  "
                     f"min_w={weights.min():.3f}  "
                     f"effective_N={1/np.sum(weights**2):.1f}")
        return self

    def rolling(self, returns: pd.DataFrame, window: int = 252,
                 step: int = 21) -> tuple[pd.DataFrame, pd.Series]:
        """
        Rolling HRP backtest.
        Returns (weights_history, portfolio_returns).
        """
        T = len(returns)
        all_weights = []
        port_dates  = []
        port_rets   = []
        w_curr = np.ones(len(returns.columns)) / len(returns.columns)

        for end in range(window, T, step):
            ret_w = returns.iloc[max(0, end-window):end]
            try:
                self.fit(ret_w)
                w_curr = self.weights_.reindex(returns.columns).fillna(0).values
            except Exception:
                pass

            hold_end = min(T, end + step)
            for t in range(end, hold_end):
                port_rets.append(returns.values[t] @ w_curr)
            all_weights.append(w_curr.copy())
            port_dates.append(returns.index[end])

        w_hist = pd.DataFrame(all_weights, index=port_dates, columns=returns.columns)
        p_ret  = pd.Series(port_rets, index=returns.index[window:T], name="HRP")
        return w_hist, p_ret


# ─────────────────────────────────────────────────────────────────────────────
class NetworkRiskMonitor:
    """
    Monitor systemic risk through dynamic network statistics.
    Rising average correlation and density signal elevated systemic risk.
    """

    def __init__(self, window: int = 126, step: int = 21, min_corr: float = 0.10):
        self.window   = window
        self.step     = step
        self.min_corr = min_corr
        self.history_: pd.DataFrame | None = None

    def compute(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling network statistics."""
        T = len(returns)
        rows = []
        for end in range(self.window, T, self.step):
            ret_w = returns.iloc[end-self.window:end]
            cn = CorrelationNetwork(min_corr=self.min_corr)
            cn.fit(ret_w)
            stats = cn.network_stats()
            stats["Date"] = returns.index[end]
            rows.append(stats)
        self.history_ = pd.DataFrame(rows).set_index("Date")
        return self.history_

    def systemic_risk_score(self) -> pd.Series:
        """
        Composite systemic risk score (0–1):
        Weighted combination of normalised avg_correlation and density.
        """
        if self.history_ is None:
            raise RuntimeError("Call compute() first.")
        h = self.history_[["avg_correlation", "density"]].copy()
        h_norm = (h - h.min()) / (h.max() - h.min() + 1e-8)
        score = 0.6 * h_norm["avg_correlation"] + 0.4 * h_norm["density"]
        return score.rename("systemic_risk_score")

    def alert_threshold(self, threshold: float = 0.80) -> pd.Series:
        """Flag dates where systemic risk score exceeds threshold."""
        score = self.systemic_risk_score()
        return score[score >= threshold]


# ─────────────────────────────────────────────────────────────────────────────
def main():
    import yfinance as yf
    MACS = ["SPY","QQQ","IWM","EFA","EEM","VNQ","AGG","TLT",
            "HYG","EMB","TIP","LQD","GLD","SLV","DBC"]
    px = yf.download(MACS, start="2015-01-01", end="2026-05-01",
                      auto_adjust=True, progress=False)["Close"].dropna()
    ret = px.pct_change().dropna()

    # Network analysis
    cn = CorrelationNetwork(min_corr=0.15)
    cn.fit(ret)
    print("Network stats:", cn.network_stats())
    print("\nTop 5 by Betweenness:")
    print(cn.centrality().nlargest(5, "Betweenness")[["Degree","Betweenness","PageRank"]].round(4))
    print("\nCommunities:")
    print(cn.communities(n_clusters=4).to_string(index=False))

    # HRP
    hrp = HRPPortfolio()
    hrp.fit(ret)
    print("\nHRP Weights:")
    print(hrp.weights_.sort_values(ascending=False).round(4).to_string())

    # Systemic risk monitor
    nrm = NetworkRiskMonitor(window=126, step=21)
    stats = nrm.compute(ret)
    score = nrm.systemic_risk_score()
    high_risk = nrm.alert_threshold(0.75)
    print(f"\nHigh systemic risk periods: {len(high_risk)} dates")
    if len(high_risk):
        print(high_risk.to_string())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
