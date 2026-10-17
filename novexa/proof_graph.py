"""Provenance log. Nodes are added only by code that actually ran."""
import time, html
class ProofGraph:
    def __init__(self): self.nodes, self.edges, self._n = [], [], 0
    def add(self, kind, label, data=None, parent=None, status="ok"):
        self._n += 1; nid = f"n{self._n}"
        self.nodes.append({"id": nid, "kind": kind, "label": label, "data": data or {}, "status": status, "t": time.time()})
        if parent: self.edges.append((parent, nid))
        if len(self.nodes) > 500: self.nodes.pop(0)
        return nid
    def to_dot(self):
        col = {"ok": "#8b7cf6", "error": "#e5484d", "warning": "#f5a524", "verified": "#30a46c", "partial": "#f5a524", "unverified": "#e5484d"}
        L = ['digraph G{rankdir=LR;bgcolor="transparent";node[shape=box,style="rounded,filled",fontname="Helvetica",fontcolor="#14141c"];edge[color="#8b7cf6"];']
        for n in self.nodes:
            lab = html.escape(f"{n['kind']}\\n{n['label'][:40]}", quote=True).replace("&amp;#92;", "\\")
            L.append(f'{n["id"]}[label="{lab}",fillcolor="{col.get(n["status"], "#8b7cf6")}"];')
        L += [f"{a}->{b};" for a, b in self.edges if any(x["id"] == a for x in self.nodes)]
        return "".join(L) + "}"
    def export(self): return {"nodes": self.nodes, "edges": self.edges}
