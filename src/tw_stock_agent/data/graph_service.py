"""供應鏈圖：NetworkX DiGraph + BFS 受益股擴張。"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

import networkx as nx

from tw_stock_agent.config import COMPANIES_JSON, cfg


class GraphService:
    """供應鏈有向圖，支援 BFS 找受益股。

    節點：公司 nickname（例如 "川湖"、"TSMC"）
    邊屬性：relation_type, confidence, strength, description
    """

    def __init__(self, path: Path | None = None):
        self._path = path or COMPANIES_JSON
        self.graph: nx.DiGraph = nx.DiGraph()
        self._reload()

    def _reload(self) -> None:
        if not self._path.exists():
            return
        blob = json.loads(self._path.read_text(encoding="utf-8"))
        companies = {c["nickname"]: c for c in blob.get("companies", [])}
        # 建節點
        for nick, c in companies.items():
            self.graph.add_node(nick, **{k: v for k, v in c.items() if k != "nickname"})
        # 建邊
        for rel in blob.get("relationships", []):
            src = rel.get("source_node", "")
            tgt = rel.get("target_node", "")
            if not src or not tgt:
                continue
            self.graph.add_edge(
                src, tgt,
                relation_type=rel.get("relation_type", "SUPPLIES_TO"),
                confidence=float(rel.get("confidence", 0.8)),
                strength=int(rel.get("strength", 5)),
                description=rel.get("description", ""),
            )

    def reload(self) -> None:
        self.graph = nx.DiGraph()
        self._reload()

    # ── BFS ────────────────────────────────────────────────────────────────

    def bfs_beneficiaries(
        self,
        seeds: list[str],
        hops: int | None = None,
        min_confidence: float | None = None,
        relation_filter: set[str] | None = None,
    ) -> list[dict]:
        """從 seed 節點做 BFS，找出供應鏈受益股。

        Args:
            seeds:  起點公司 nickname list（例如來自新聞的 ["Nvidia", "AMD"]）
            hops:   最大跳數（預設 yaml 設定）
            min_confidence: 最低邊信心度
            relation_filter: 只走哪些關係類型（None = 全部）

        Returns:
            List of dicts: {nickname, code, yf_ticker, bfs_depth, via_path, source_seeds}
        """
        max_hops = hops if hops is not None else cfg("bfs.max_hops", 3)
        min_conf = min_confidence if min_confidence is not None else cfg("bfs.min_confidence", 0.6)

        visited: dict[str, int] = {}  # nickname → depth first found
        queue: deque[tuple[str, int, list[str]]] = deque()

        for s in seeds:
            if s in self.graph:
                queue.append((s, 0, [s]))
                visited[s] = 0

        results: list[dict] = []

        while queue:
            node, depth, path = queue.popleft()
            if depth >= max_hops:
                continue
            for neighbor in self.graph.successors(node):
                edge = self.graph[node][neighbor]
                if edge.get("confidence", 1.0) < min_conf:
                    continue
                if relation_filter and edge.get("relation_type") not in relation_filter:
                    continue
                if neighbor not in visited:
                    visited[neighbor] = depth + 1
                    new_path = path + [neighbor]
                    queue.append((neighbor, depth + 1, new_path))
                    # 只收台股（有 code 且是純數字）
                    node_data = self.graph.nodes.get(neighbor, {})
                    code = node_data.get("code", "")
                    if code and code.isdigit():
                        market = node_data.get("country", "TW")
                        suffix = ".TW" if market == "TW" else ".TWO"
                        results.append({
                            "nickname": neighbor,
                            "name": node_data.get("name_zh", neighbor),
                            "code": code,
                            "yf_ticker": f"{code}{suffix}",
                            "bfs_depth": depth + 1,
                            "via_path": " → ".join(new_path),
                            "source_seeds": [s for s in seeds if s in path],
                        })

        # 去重，保留最淺的
        seen: dict[str, dict] = {}
        for r in results:
            c = r["code"]
            if c not in seen or r["bfs_depth"] < seen[c]["bfs_depth"]:
                seen[c] = r
        return list(seen.values())

    def get_code_to_nickname(self) -> dict[str, str]:
        """Return {code: nickname} for all graph nodes that have a numeric code."""
        return {
            data.get("code", ""): node
            for node, data in self.graph.nodes(data=True)
            if data.get("code")
        }

    def get_node_info(self, nickname: str) -> dict:
        return dict(self.graph.nodes.get(nickname, {}))

    def all_tw_tickers(self) -> list[dict]:
        """列出圖中所有台股節點（作為預備的 universe）。"""
        out = []
        for node, data in self.graph.nodes(data=True):
            code = data.get("code", "")
            if code and code.isdigit():
                out.append({"nickname": node, "code": code,
                            "name": data.get("name_zh", node),
                            "yf_ticker": f"{code}.TW"})
        return out
