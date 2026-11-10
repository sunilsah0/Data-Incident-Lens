import json
from engine.store import get_store, init_store

MANIFEST = "pipelines/target/manifest.json"


def criticality(name, asset_type):
    if asset_type == "source":
        return 1
    if name.startswith("mart_"):
        return 3      # business-facing tables matter most
    return 2


def collect_lineage():
    init_store()
    with open(MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)

    assets = {}   # asset_id -> (type, name, criticality)
    edges = set()

    for key, src in manifest["sources"].items():
        asset_id = f'{src["schema"]}.{src["name"]}'
        assets[asset_id] = ("source", src["name"], criticality(src["name"], "source"))

    id_of = {}
    for key, src in manifest["sources"].items():
        id_of[key] = f'{src["schema"]}.{src["name"]}'

    for key, node in manifest["nodes"].items():
        if node["resource_type"] != "model":
            continue
        asset_id = f'{node["schema"]}.{node["name"]}'
        id_of[key] = asset_id
        assets[asset_id] = ("model", node["name"], criticality(node["name"], "model"))

    for key, node in manifest["nodes"].items():
        if node["resource_type"] != "model":
            continue
        for dep in node["depends_on"]["nodes"]:
            if dep in id_of:
                edges.add((id_of[dep], id_of[key]))

    con = get_store()
    con.execute("DELETE FROM assets")
    con.execute("DELETE FROM lineage_edges")
    for asset_id, (atype, name, crit) in assets.items():
        con.execute("INSERT INTO assets VALUES (?, ?, ?, ?)", [asset_id, atype, name, crit])
    for up, down in edges:
        con.execute("INSERT INTO lineage_edges VALUES (?, ?)", [up, down])
    con.close()
    print(f"Saved {len(assets)} assets and {len(edges)} lineage edges")


if __name__ == "__main__":
    collect_lineage()