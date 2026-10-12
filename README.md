# agent-cad

エージェント駆動で製造品質のCADモデルを開発するためのリポジトリ。
カーネルは OpenCascade（build123d / OCP）、出力は STEP AP242。

## 構成

- `agent_cad/parts/` パラメトリック部品（寸法・材質・質量目標を引数で変更可能）
  - `stem.py` 自転車ステム（2ボルト・コラムクランプ、4ボルト・フェイスプレート）
- `agent_cad/fea/` 線形静解析（gmsh + scikit-fem, P2四面体）
- `agent_cad/step_io.py` STEP AP242 入出力（製品構成・部品名・色・材質/密度・幾何検証プロパティ付き）
- `agent_cad/verify/step_check.py` STEP受け渡しの自己完結チェック（下記）
- `agent_cad/drawing/` JIS図面（第三角法、断面・矢視、寸法・はめあい・幾何公差・表面粗さ、表題欄）をPDF/DXFで出力
- `agent_cad/standards/jis.py` JIS B 0405 普通公差、JIS B 0401 はめあい表
- `agent_cad/parts/stem_drawing.py` ステム本体・フェイスプレートの図面定義
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
| geom | 体積・表面積・重心が元形状と一致し、ファイル内の幾何検証プロパティとも一致 |
| mesh | 独立した三角形分割の体積がB-rep体積と0.5%以内（面欠落の検出） |
| tol | 最大トレランス ≤ 0.01 mm、最短エッジ ≥ 0.01 mm（CATIAの分解能0.001 mmに対し余裕） |
| rewrite | 読み込んだモデルを書き直して再読込しても質量特性が不変 |

幾何検証プロパティ（体積・面積・重心）はAP242の仕組みで、CATIA側でも読込時に照合できる。
材質名と密度もSTEPに書き込む。

```sh
python -m pytest -q tests   # 正常ファイルの合格と、面欠落・検証値改ざん・単位違いの検出を確認
```

## 図面（JIS）

`make_stem` は部品ごとに `out/<名前>_<部品>_drawing.pdf` と `.dxf` を出力する。

- 第三角法、A3、尺度は1:1（小物は2:1）。収まらなければ尺度を下げ、それでも収まらなければA2
- 図番は部品パラメータのハッシュから付ける（`AC-STEM-xxxxxx-01`）。寸法・角度・材料が違う品番は必ず別図番になる
- PDFには日本語フォントが必要（IPAゴシック、Noto Sans CJK JPなど。例: `apt-get install fonts-ipafont-gothic`）。見つからないときは文字化けしたPDFを出さずにエラーで止まる
- 寸法値はパラメータではなくB-repから測った値を記入する（穴径、軸間距離、ボルトピッチなど）
- はめあい公差はJIS B 0401の表から計算する（クランプ穴はF8）。普通公差はJIS B 0405-m、幾何公差の普通公差はJIS B 0419-K
- 表面粗さはJIS B 0031の記号で記入する（一括指示はRa 3.2、クランプ穴と合わせ面はRa 1.6）
- 材料はJIS H 4140の鍛造品記号、表面処理はJIS H 8601

図面チェック（`agent_cad/drawing/check.py`）：

- 測った寸法と設計意図（パラメータ）が0.005以内で一致すること
- はめあいの許容差が表の値と一致すること
- 表題欄・投影法・普通公差・一括粗さの記載があること
- 文字どうし、および文字と外形線が重ならず、文字が枠内に収まること
