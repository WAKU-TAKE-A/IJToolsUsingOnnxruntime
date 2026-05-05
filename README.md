# IJToolsUsingOnnxruntime

ONNX Runtime を推論エンジンとして使用する ImageJ プラグイン群です。  
YOLO ベースの物体検出・ポーズ推定・画像分類に対応しています。

## 特徴

- **高速推論**: Microsoft ONNX Runtime for Java を採用。
- **YOLO 対応**:
  - 物体検出（通常モデル・E2E モデル）
  - ポーズ推定（通常モデル・E2E モデル）
  - 画像分類
- **標準 UI**: ImageJ の `GenericDialog` と `DialogListener` を使用したインタラクティブなダイアログ。
- **マクロ互換**: ImageJ マクロレコーダーおよびヘッドレス実行に完全対応。
- **モデル検査**: ONNX モデルのメタデータ・入出力構造を確認するユーティリティを内蔵。

## インストール

1. Maven でビルドします：
   ```powershell
   mvn clean package -DskipTests
   ```
2. 生成された `target/IJTools_UsingOnnxruntime.jar` を ImageJ の `plugins` フォルダにコピーします。
3. ImageJ を再起動します。

## 使い方

### 1. モデルの読み込み
`Plugins > ORT > 1st Read and Release` を開き、`.onnx` ファイルと形式（例：`YOLO_Object_Pixel`、`YOLO_Pose`）を指定します。

### 2. 推論の実行
画像を開いてから、以下のメニューを選択します：
- `Plugins > ORT > 2nd Detection` — 物体検出
- `Plugins > ORT > 2nd Pose` — ポーズ推定
- `Plugins > ORT > 2nd Classify` — 画像分類

### 3. モデルの検査
`Plugins > ORT > Inspect` で、ONNX ファイルの入出力ノードやメタデータを確認できます。

## 開発者向け

`test/` および `yolo/` フォルダにユーティリティが含まれています：
- `test/test_all_models.ijm` — 全モデルを対象とした自動テストマクロ
- `yolo/inspect_onnx.py` — CLI でモデル構造を検査するスクリプト
- `yolo/extract_names.py` — モデルのメタデータからクラス名を抽出するスクリプト

CLI でのテスト方法の詳細は `test/README_CLI_Testing.md` を参照してください。

## ライセンス
MIT License
