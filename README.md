# IJToolsUsingOnnxruntime

ONNX Runtime を推論エンジンとして使用する ImageJ プラグイン群です。
YOLO ベースの物体検出・ポーズ推定・画像分類・セグメンテーション・OBB（回転矩形）検出に幅広く対応しています。
Roboflow RF-DETR の物体検出・インスタンスセグメンテーション・姿勢推定（Keypoint Preview）にも対応します。

RF-DETR のモデル取得・サンプル画像・検証手順は [RF-DETR/README.md](RF-DETR/README.md) を参照してください。
Python の作業環境は `RF-DETR/.venv`、モデルの保存先は `RF-DETR/models/` です。
YOLO の補助スクリプトは [yolo/README.md](yolo/README.md)、エージェント向けの構成・注意点・検証手順は [README_AI.md](README_AI.md) を参照してください。

## 特徴

- **高速推論**: Microsoft ONNX Runtime for Java を採用。
- **YOLO 全タスク対応**:
  - 物体検出 (YOLO Detect) - 通常モデル・E2Eモデル
  - ポーズ推定 (YOLO Pose) - 通常モデル・E2Eモデル
  - 画像分類 (YOLO Classify)
  - インスタンスセグメンテーション (YOLO Segment) - 通常モデル・E2Eモデル
  - 回転矩形検出 (YOLO OBB) - 通常モデル・E2Eモデル
- **標準UI**: ImageJ の `GenericDialog` と `DialogListener` を使用したインタラクティブなプレビュー表示。
- **マクロ互換**: ImageJ マクロレコーダーおよびヘッドレス実行に対応。
- **モデル検査**: ONNX モデルのメタデータ・入出力構造を確認するユーティリティを内蔵。

## インストール

1. Maven でビルドします。
   ```bash
   mvn clean package
   ```
2. 生成された `target/IJTools_UsingOnnxruntime.jar` を ImageJ の `plugins` フォルダにコピーします。
3. ImageJ を再起動します。

## 使い方

### 1. モデルの読み込み
`Plugins > ORT > 1st Read and Release` を開き、`action=read_model`、`.onnx` ファイル、形式（例：`YOLO_OBB_E2E`、`YOLO_Segment` 等）、スロットを指定して読み込みます。

### 2. 推論の実行
対象の画像を開いてから、以下の中から読み込んだモデルに合ったメニューを選択します。
- `Plugins > ORT > 2nd Detection` （物体検出）
- `Plugins > ORT > 2nd Pose` （ポーズ推定）
- `Plugins > ORT > 2nd Classify` （画像分類）
- `Plugins > ORT > 2nd Segmentation` （セグメンテーション）
- `Plugins > ORT > 2nd OBB` （回転矩形検出）

### 3. モデルの検査
`Plugins > ORT > Inspect` で、ONNX ファイルの入出力ノードやメタデータ（学習時画像サイズ、クラス名など）を確認できます。

## ライセンス

本プロジェクトは [MIT License](LICENSE) です。
配布 JAR には、本プロジェクトのライセンスを `META-INF/LICENSE`、ONNX Runtime のライセンスを
`META-INF/licenses/onnxruntime-LICENSE.txt` に同梱し、依存 JAR の `ThirdPartyNotices.txt` も保持します。
同梱内容の案内は `META-INF/NOTICE.txt` にあります。
