# agent-cad

エージェント駆動で製造品質のCADモデルを開発するためのリポジトリ。
カーネルは OpenCascade（build123d / OCP）、出力は STEP AP242。

## 構成

- `agent_cad/parts/` パラメトリック部品（寸法・材質・質量目標を引数で変更可能）
  - `stem.py` 自転車ステム（2ボルト・コラムクランプ、4ボルト・フェイスプレート）
- `agent_cad/fea/` 線形静解析（gmsh + scikit-fem, P2四面体）
- `agent_cad/step_io.py` STEP AP242 入出力（製品構成・部品名・色・材質/密度・幾何検証プロパティ付き）
- `agent_cad/verify/step_check.py` STEP受け渡しの自己完結チェック（下記）
- `agent_cad/render.py` 簡易レンダリング（matplotlib）
- `scripts/` 生成スクリプト（出力は `out/`）

## 使い方

```sh
pip install -r requirements.txt
python -m scripts.make_stem          # STEP / レポート / 画像を out/ に出力
```

## STEP受け渡しチェック（CATIA読込テストの代替）

CATIA実機での読込確認ができるまで、書き出したSTEPを以下で自動検証する。`make_stem` は
各STEPに対して実行し、結果を `out/*_report.json` の `step_exchange` に入れる。

| 区分 | 内容 |
|---|---|
| text | OpenCascadeを使わない独立パース：スキーマがAP242、長さ単位mm、全参照が解決、ソリッド数 |
| struct | 製品構成（アセンブリ名、部品名）と色が往復で保たれる |
| topo | 各部品が1つの有効な閉ソリッド（自由エッジ0） |
| geom | 体積・表面積・重心が元形状と一致し、ファイル内の幾何検証プロパティとも一致。慣性テンソル・外形範囲・表面上の点の照合で向きや配置のずれも検出 |
| mesh | 独立した三角形分割の体積がB-rep体積と0.5%以内（面欠落の検出） |
| tol | 最大トレランス ≤ 0.01 mm、最短エッジ ≥ 0.01 mm（CATIAの分解能0.001 mmに対し余裕） |
| rewrite | 読み込んだモデルを書き直して再読込しても質量特性が不変 |

幾何検証プロパティ（体積・面積・重心）はAP242の仕組みで、CATIA側でも読込時に照合できる。
材質名と密度もSTEPに書き込む。

```sh
python -m pytest -q tests   # 正常ファイルの合格と、面欠落・検証値改ざん・単位違いの検出を確認
```
