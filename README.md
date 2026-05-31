# IJToolsUsingOnnxruntime

ONNX Runtime を推論エンジンとして使用する ImageJ プラグイン群です。
YOLO ベースの物体検出・ポーズ推定・画像分類・セグメンテーション・OBB（回転矩形）検出に幅広く対応しています。

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
`Plugins > ORT > 1st Read and Release` を開き、`.onnx` ファイルと形式（例：`YOLO_OBB_E2E`、`YOLO_SEG` 等）を指定して「Load」します。

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
MIT License
