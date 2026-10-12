# agent-cad

エージェント駆動で製造品質のCADモデルを開発するためのリポジトリ。
カーネルは OpenCascade（build123d / OCP）、出力は STEP AP242。

## 構成

- `agent_cad/parts/` パラメトリック部品（寸法・材質・質量目標を引数で変更可能）
  - `stem.py` 自転車ステム（2ボルト・コラムクランプ、4ボルト・フェイスプレート）
- `agent_cad/fea/` 線形静解析（gmsh + scikit-fem, P2四面体）
- `agent_cad/render.py` 簡易レンダリング（matplotlib）
- `scripts/` 生成スクリプト（出力は `out/`）

## 使い方

```sh
pip install -r requirements.txt
python -m scripts.make_stem          # STEP / レポート / 画像を out/ に出力
```
